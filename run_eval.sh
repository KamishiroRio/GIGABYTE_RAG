#!/bin/bash

# 1. 啟動虛擬環境
source .venv/bin/activate

# 2. 掛載 CUDA 相關的底層函式庫路徑 (解決 libcudart 找不到的問題)
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusparse/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

# 3. 進入 code 目錄並執行評測腳本
echo "啟動自動化評測管線..."
cd code
python3 auto_eval.py