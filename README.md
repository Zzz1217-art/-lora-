# Qwen-VL-7B LoRA 微调

这个目录是一套不依赖 LLaMA-Factory 的最小 LoRA SFT 代码，适合在训练主机上运行。你可以在 Mac 上改代码，然后把整个 `qwen_vl_lora` 文件夹拷到 Windows 训练主机运行。

## 1. 安装依赖

```bash
cd /Users/apple/Documents/code/qwen_vl_lora
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果训练机是 Windows + NVIDIA GPU，建议先按你的 CUDA 版本安装 PyTorch，再安装本文件里的其他依赖。4bit 量化依赖 `bitsandbytes`，如果原生 Windows 安装或运行报错，最稳的是在 WSL2/Linux CUDA 环境运行同一套代码。

## 2. 数据格式

支持 JSON 数组，例如：

```json
[
  {
    "image": "001.jpg",
    "question": "图片里有什么？",
    "answer": "图片里有一只杯子。"
  }
]
```

也兼容 `image_path/img/picture`、`prompt/query/instruction/input`、`response/output/target`，以及常见的 `conversations/messages` 格式。

## 3. 开始训练

Windows 上最简单的方式是改 `run_train_windows.py` 顶部这几行：

```python
MODEL_PATH = r"D:\models\Qwen-VL-7B"
DATA_PATH = r"D:\datasets\my_vl_dataset\train.json"
IMAGE_ROOT = r"D:\datasets\my_vl_dataset\images"
OUTPUT_DIR = r"D:\outputs\qwen-vl-7b-lora"
```

4bit、thinking 和视觉编码器开关也在同一个文件里：

```python
USE_4BIT = True
DISABLE_THINKING = True
TRAIN_VISION_ENCODER = True
```

然后运行：

```bash
cd D:\path\to\qwen_vl_lora
python run_train_windows.py
```

也可以直接运行命令：

```bash
python train_qwen_vl_lora.py \
  --model_name_or_path /path/to/Qwen-VL-7B \
  --data_path /path/to/train.json \
  --image_root /path/to/images \
  --output_dir /path/to/qwen-vl-7b-lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --max_length 2048 \
  --fp16 \
  --gradient_checkpointing \
  --use_4bit \
  --disable_thinking \
  --train_vision_encoder
```

Linux CUDA 显存紧张时可以加：

```bash
  --use_4bit
```

4bit 量化用 `--use_4bit`。如果 Windows 原生环境的 `bitsandbytes` 报错，代码不用改，换到 WSL2/Linux CUDA 环境执行即可。

`--disable_thinking` 会移除答案里的 `<think>...</think>` 片段，并在用户问题后加一句“不要输出思考过程”的约束。Qwen-VL-7B 本身通常不是 thinking 模型，这个开关主要用于你的数据里混入了 thinking 标签，或你想强制只学最终答案。

视觉编码器默认会参与图片理解。`--train_vision_encoder` 表示视觉编码器参数也参与训练，显存占用会明显增加；如果只是普通 LoRA，通常可以不开它。

## 4. 推理验证

Windows 上可以改 `run_infer_windows.py` 顶部路径后运行：

```bash
python run_infer_windows.py
```

```bash
python infer_lora.py \
  --model_name_or_path /path/to/Qwen-VL-7B \
  --lora_path /path/to/qwen-vl-7b-lora \
  --image /path/to/images/001.jpg \
  --question "图片里有什么？" \
  --fp16
```

## 5. 常用调整

- 显存不够：降低 `--max_length`，保持 batch size 为 1，增大或保持 `--gradient_accumulation_steps`，打开 `--gradient_checkpointing`，使用 `--use_4bit`，必要时关闭 `TRAIN_VISION_ENCODER`。
- 过拟合：降低 epoch 或 LoRA rank，例如 `--lora_r 16`。
- 训练太慢：增大 batch size，前提是显存允许。
- LoRA 目标模块报错：先用默认值；如果你的 Qwen-VL 版本模块名不同，可改 `--lora_target_modules`。
