#!/bin/bash

# CosyVoice WebUI API (headless, 8000) 启动脚本

echo "正在启动 CosyVoice API (8000)..."

# 获取 conda 路径（直接使用已知的 Miniconda 安装位置）
CONDA_BASE="/home/boom/miniconda"

# 激活 conda 环境
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cosyvoice310

# 运行 Python 脚本，禁用输出缓冲
python -u cosyvoice_api.py
