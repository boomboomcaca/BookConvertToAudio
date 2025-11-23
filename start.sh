#!/bin/bash

# CosyVoice 有声书转换器启动脚本

echo "正在启动 CosyVoice Web UI..."

# 获取 conda 路径（直接使用已知的 Miniconda 安装位置，避免依赖全局 conda 命令）
CONDA_BASE="/home/boom/miniconda3"

# 激活 conda 环境
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cosyvoice

# 运行 Python 脚本，禁用输出缓冲
python -u web_book_converter.py
