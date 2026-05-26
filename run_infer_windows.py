import subprocess
import sys


MODEL_PATH = r"D:\models\Qwen-VL-7B"
LORA_PATH = r"D:\outputs\qwen-vl-7b-lora"
IMAGE_PATH = r"D:\datasets\my_vl_dataset\images\001.jpg"
QUESTION = "请描述这张图片。"

USE_FP16 = True
USE_BF16 = False


def main() -> None:
    cmd = [
        sys.executable,
        "infer_lora.py",
        "--model_name_or_path",
        MODEL_PATH,
        "--lora_path",
        LORA_PATH,
        "--image",
        IMAGE_PATH,
        "--question",
        QUESTION,
    ]

    if USE_FP16:
        cmd.append("--fp16")
    if USE_BF16:
        cmd.append("--bf16")

    print("Running command:")
    print(" ".join(f'"{item}"' if " " in item else item for item in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
