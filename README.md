## Overview
This is an QA AI assistant system that answers spec. about AORUS MASTER 16 AM6H. This system runs under limited computing resources.

### Tech stack
- Environment Management: `uv`
- Inference Engine: `llama.cpp`
- Search System: `Python Vector DB` based on `numpy` and `SentenceTransformers`
- Hardware limitation: VRAM usage \< 4GB

### Testing environment
- OS: Windows 11 with WSL 2 (Ubuntu)
    - WSL version：2.7.13.0
    - Core version：6.18.33.2-2
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)
- RAM: 16GB
- CPU: Intel Core i7-11800H


## Quick Start

### 1. Clone Repo
```bash
git clone https://github.com/KamishiroRio/GIGABYTE_RAG
cd GIGABYTE_RAG
```

### 2. Set up venv
```bash
# Install uv (if not)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Set up env
uv venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
# install prerequisties
uv pip install requests beautifulsoup4 numpy sentence-transformers huggingface_hub

# install llama-cpp-python inference engine that supports CUDA 12 
uv pip install llama-cpp-python --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cu124](https://abetlen.github.io/llama-cpp-python/whl/cu124)
```

### 4. Download models
```bash
cd code
mkdir -p ../models
# download model
hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ../models
```

### 5. Run the QA assistant
```bash
python3 rag_app.py
``` 

## Main Files Introduction
- ./code/AORUS MASTER 16 AM6H 筆記型電腦 產品規格 - GIGABYTE 技嘉科技.html: HTML file that directly downloaded from the given link: https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp
- ./code/data_parser.py: Extract the data from the html file. Each chunk has the form `{model_name} 的{title}為：{val_text}`, such as `AORUS MASTER 16 BZH 的作業系統為：Windows 11 Pro (GIGABYTE recommends Windows 11 Pro for business.) Windows 11 Home UEFI Shell OS。`
- ./code/vector_search.py: Helper functions of vector DB which include `add_text()` and `search()`. This can be also directly executed to test cosine similarities.
- ./code/rag_app.py: The main script of the QA system, which includes VRAM monitoring and evaluation.
