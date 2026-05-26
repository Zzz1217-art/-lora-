import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a Qwen-VL LoRA adapter.")
    parser.add_argument("--model_name_or_path", required=True, help="Base Qwen-VL-7B path.")
    parser.add_argument("--lora_path", required=True, help="LoRA adapter output path.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    query = tokenizer.from_list_format([{"image": args.image}, {"text": args.question}])
    response, _ = model.chat(tokenizer, query=query, history=None)
    print(response)


if __name__ == "__main__":
    main()
