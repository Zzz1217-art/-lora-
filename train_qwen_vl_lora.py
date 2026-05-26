import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


IGNORE_INDEX = -100
IMAGE_KEYS = ("image", "image_path", "img", "picture")
QUESTION_KEYS = ("question", "query", "instruction", "prompt", "input")
ANSWER_KEYS = ("answer", "response", "output", "target")


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "records", "annotations"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Dataset must be a JSON list or a dict containing data/samples/records/annotations.")


def first_existing(record: Dict[str, Any], keys: tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_image_path(image: Any, image_root: Optional[str]) -> str:
    if isinstance(image, list):
        if not image:
            raise ValueError("Empty image list in record.")
        image = image[0]
    if not isinstance(image, str):
        raise ValueError(f"Image path must be a string, got: {type(image)}")

    image_path = Path(image).expanduser()
    if not image_path.is_absolute() and image_root:
        image_path = Path(image_root).expanduser() / image_path
    return str(image_path)


def text_from_messages(messages: List[Dict[str, Any]], role_names: tuple[str, ...]) -> str:
    parts: List[str] = []
    for message in messages:
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in role_names:
            value = message.get("content") or message.get("value") or ""
            if isinstance(value, list):
                value = " ".join(str(item.get("text", item)) for item in value)
            parts.append(str(value).replace("<image>", "").strip())
    return "\n".join(part for part in parts if part).strip()


def parse_record(record: Dict[str, Any], image_root: Optional[str]) -> Dict[str, str]:
    image = first_existing(record, IMAGE_KEYS)
    if image is None and isinstance(record.get("images"), list) and record["images"]:
        image = record["images"][0]
    if image is None:
        raise ValueError(f"Missing image path in record: {record}")

    question = first_existing(record, QUESTION_KEYS)
    answer = first_existing(record, ANSWER_KEYS)

    conversations = record.get("conversations") or record.get("messages")
    if isinstance(conversations, list):
        question = question or text_from_messages(conversations, ("human", "user"))
        answer = answer or text_from_messages(conversations, ("gpt", "assistant"))

    if not question or not answer:
        raise ValueError(f"Missing question/answer text in record: {record}")

    return {
        "image": normalize_image_path(image, image_root),
        "question": str(question).strip(),
        "answer": str(answer).strip(),
    }


def remove_thinking_text(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            break
        text = text[:start] + text[end + len("</think>") :]
    return text.strip()


class QwenVlSftDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        image_root: Optional[str],
        max_length: int,
        disable_thinking: bool,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.disable_thinking = disable_thinking
        self.records = [parse_record(item, image_root) for item in load_json_records(data_path)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.records[index]
        query = self.tokenizer.from_list_format(
            [
                {"image": item["image"]},
                {"text": item["question"]},
            ]
        )
        if self.disable_thinking:
            query = f"{query}\n请直接给出最终答案，不要输出思考过程或 <think> 内容。"
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        answer = remove_thinking_text(item["answer"]) if self.disable_thinking else item["answer"]
        full_text = f"{prompt}{answer}<|im_end|>"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        }


@dataclass
class DataCollatorForQwenVl:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eod_id if hasattr(self.tokenizer, "eod_id") else self.tokenizer.eos_token_id

        max_len = max(feature["input_ids"].size(0) for feature in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            length = feature["input_ids"].size(0)
            pad_len = max_len - length
            batch["input_ids"].append(torch.cat([feature["input_ids"], torch.full((pad_len,), pad_id)]))
            batch["labels"].append(torch.cat([feature["labels"], torch.full((pad_len,), IGNORE_INDEX)]))
            batch["attention_mask"].append(torch.cat([feature["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))

        return {key: torch.stack(value) for key, value in batch.items()}


def set_vision_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    visual_modules = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if lowered.endswith("visual") or "visual_encoder" in lowered or "vision_tower" in lowered:
            visual_modules.append((name, module))

    if not visual_modules:
        print("Warning: no explicit vision encoder module name was found. The model will still use images in forward.")
        return

    for name, module in visual_modules:
        for param in module.parameters():
            param.requires_grad = trainable
        state = "trainable" if trainable else "frozen"
        print(f"Vision encoder module '{name}' is {state}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen-VL-7B without LLaMA-Factory.")
    parser.add_argument("--model_name_or_path", required=True, help="Local Qwen-VL-7B model path.")
    parser.add_argument("--data_path", required=True, help="JSON dataset path.")
    parser.add_argument("--image_root", default=None, help="Root directory for relative image paths.")
    parser.add_argument("--output_dir", default="./qwen-vl-7b-lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="c_attn,attn.c_proj,w1,w2",
        help="Comma separated module names. Common Qwen targets: c_attn,attn.c_proj,w1,w2.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_4bit", action="store_true", help="Use QLoRA 4-bit loading.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true", help="Remove <think> blocks and ask the model for final answers only.")
    parser.add_argument("--train_vision_encoder", action="store_true", help="Make the vision encoder trainable.")
    parser.add_argument("--freeze_vision_encoder", action="store_true", help="Freeze the vision encoder parameters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_vision_encoder and args.freeze_vision_encoder:
        raise ValueError("Use only one of --train_vision_encoder or --freeze_vision_encoder.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        quantization_config=quantization_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[name.strip() for name in args.lora_target_modules.split(",") if name.strip()],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if args.train_vision_encoder:
        set_vision_encoder_trainable(model, True)
    if args.freeze_vision_encoder:
        set_vision_encoder_trainable(model, False)
    model.print_trainable_parameters()

    train_dataset = QwenVlSftDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        image_root=args.image_root,
        max_length=args.max_length,
        disable_thinking=args.disable_thinking,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForQwenVl(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
