import subprocess
import sys


# 只需要改这里：模型、数据集、图片文件夹、输出目录
MODEL_PATH = r"D:\models\Qwen-VL-7B"
DATA_PATH = r"D:\datasets\my_vl_dataset\train.json"
IMAGE_ROOT = r"D:\datasets\my_vl_dataset\images"
OUTPUT_DIR = r"D:\outputs\qwen-vl-7b-lora"

# 常用训练参数
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = "2e-4"
MAX_LENGTH = 2048
SAVE_STEPS = 200
LOGGING_STEPS = 10

USE_FP16 = True
USE_BF16 = False
USE_4BIT = True
USE_GRADIENT_CHECKPOINTING = True
DISABLE_THINKING = True
DEBUG_LORA_TENSORS = True
DEBUG_MAX_MODULES = 8
SAVE_TRAINING_PLOTS = True
DEBUG_DIR = r"D:\outputs\qwen-vl-7b-lora\debug"

# 视觉编码器会始终参与图片理解。设为 True 表示视觉编码器参数也一起训练，显存占用会明显增加。
TRAIN_VISION_ENCODER = True
FREEZE_VISION_ENCODER = False


def main() -> None:
    cmd = [
        sys.executable,
        "train_qwen_vl_lora.py",
        "--model_name_or_path",
        MODEL_PATH,
        "--data_path",
        DATA_PATH,
        "--image_root",
        IMAGE_ROOT,
        "--output_dir",
        OUTPUT_DIR,
        "--num_train_epochs",
        str(EPOCHS),
        "--per_device_train_batch_size",
        str(BATCH_SIZE),
        "--gradient_accumulation_steps",
        str(GRAD_ACCUM),
        "--learning_rate",
        str(LR),
        "--max_length",
        str(MAX_LENGTH),
        "--save_steps",
        str(SAVE_STEPS),
        "--logging_steps",
        str(LOGGING_STEPS),
    ]

    if USE_FP16:
        cmd.append("--fp16")
    if USE_BF16:
        cmd.append("--bf16")
    if USE_4BIT:
        cmd.append("--use_4bit")
    if USE_GRADIENT_CHECKPOINTING:
        cmd.append("--gradient_checkpointing")
    if DISABLE_THINKING:
        cmd.append("--disable_thinking")
    if DEBUG_LORA_TENSORS:
        cmd.extend(["--debug_lora_tensors", "--debug_max_modules", str(DEBUG_MAX_MODULES), "--debug_dir", DEBUG_DIR])
    if SAVE_TRAINING_PLOTS:
        cmd.append("--save_training_plots")
    if TRAIN_VISION_ENCODER:
        cmd.append("--train_vision_encoder")
    if FREEZE_VISION_ENCODER:
        cmd.append("--freeze_vision_encoder")

    print("Running command:")
    print(" ".join(f'"{item}"' if " " in item else item for item in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
