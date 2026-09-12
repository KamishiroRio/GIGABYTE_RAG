## Overview
This project is a Retrieval-Augmented Generation (RAG) QA assistant for answering hardware specification questions about the GIGABYTE AORUS MASTER 16 AM6H series.

The system is designed to run under limited computing resources, with a VRAM budget of less than 4 GB.

### Tech stack
- Environment Management: `uv`
- Inference Engine: `llama.cpp`
- Search System: `Python Vector DB` based on `numpy` and `SentenceTransformers`
- Target VRAM budget: < 4 GB

### Testing environment
- OS: Windows 11 with WSL 2 (Ubuntu)
    - WSL version：2.7.13.0
    - Core version：6.18.33.2-2
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)
- RAM: 16GB
- CPU: Intel Core i7-11800H


## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/KamishiroRio/GIGABYTE_RAG
cd GIGABYTE_RAG
```

### 2. Environment Setup & Dependency Sync
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies
uv sync
```

### 3. Download the LLM Model
```bash
mkdir -p models
# Download model via huggingface-cli
uv run huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir models
```

### 4. Run the QA Assistant
No need to manually export paths. The startup script handles virtual environment activation and CUDA library linking automatically.

```bash
chmod +x run.sh
./run.sh
```

---

## Automated Evaluation Pipeline

To ensure the reliability of the RAG system, we developed a local **Hybrid Evaluation Pipeline** that tests 100 curated cases across Factual, Comparison, Ambiguous, and Multilingual query types.

To run the benchmark:
```bash
chmod +x run_eval.sh
./run_eval.sh
```
This script will output detailed hardware metrics (TTFT, TPS) and generate a comprehensive `evaluation_report.txt` and an `error_cases.txt` in the `report/` directory.

## System Pipeline

1. Parse the official HTML specification page.
2. Convert specification entries into model-aware text chunks.
3. Encode chunks using SentenceTransformers.
4. Store embeddings in a NumPy-based vector database.
5. Retrieve the top-K relevant chunks for each query.
6. Generate answers using Qwen2.5-3B-Instruct through llama.cpp.
7. Evaluate retrieval quality, answer correctness, and hardware performance separately.

## Main Files Introduction
- **./code/AORUS MASTER 16 AM6H 筆記型電腦 產品規格 - GIGABYTE 技嘉科技.html**: HTML file that directly downloaded from the given link: https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp
- **./code/data_parser.py**: Extract the data from the html file. Each chunk has the form `{model_name} 的{title}為：{val_text}`, such as `AORUS MASTER 16 BZH 的作業系統為：Windows 11 Pro (GIGABYTE recommends Windows 11 Pro for business.) Windows 11 Home UEFI Shell OS。`
- **./code/vector_search.py**: Helper functions of vector DB which include `add_text()` and `search()`. This can be also directly executed to test cosine similarities.
- **./code/rag_app.py**: The main script of the QA system, which includes VRAM monitoring and evaluation.
- **./code/auto_eval.py**: An automated hybrid evaluation pipeline combining rule-based validation, alias matching, and semantic similarity scoring for retrieval and answer quality assessment.
- **./code/test_cases.json**: 100 curated and manually reviewed test cases, initially generated with LLM assistance.

## Model Selection

To meet the 4GB VRAM limitation, the model selection should prioritize a practical balance between memory efficiency, cross-lingual comprehension, and instruction-following capability.

* **Selected Model**: `Qwen2.5-3B-Instruct`
* **Format & Quantization**: GGUF (`q4_k_m`)
* **Selection Rationale**:
  * **Memory Efficiency**: In the tested configuration, the Q4_K_M quantized 3B model fits within the 4GB VRAM budget while leaving room for runtime overhead.
  * **Multilingual Capability**: Qwen2.5 natively supports robust cross-lingual tasks. It accurately comprehends the Traditional Chinese specification sheets and reliably responds in the queried language (Chinese or English).
  * **Instruction Following**: Despite its compact size, it is capable of adhering to strict system prompts—such as outputting specific rejection phrases for out-of-context questions and structuring multi-model comparisons—which is critical for passing the automated evaluation pipeline.

### Evaluation Methodology & Scoring Mechanism

The project utilizes two separate scripts to ensure clean profiling of hardware performance and rigorous testing of response quality.

#### 1. Hardware & System Metrics (`rag_app.py`)
Hardware performance is profiled dynamically during real-time interactive inference to ensure measurements are isolated from background evaluation overhead.
* **TTFT (Time To First Token)**: Measured using Python's `time` module, recording the latency from prompt submission to the generation of the first token.
* **TPS (Tokens Per Second)**: Calculated by dividing total generated tokens by the generation duration (excluding TTFT).
* **VRAM Usage**: Monitored dynamically via `subprocess` calls to `nvidia-smi` (`--query-gpu=memory.used`), tracking the exact GPU memory footprint during active inference.

#### 2. Automated Evaluation Pipeline (`auto_eval.py`)
Response accuracy and retrieval performance are evaluated using a custom 100-case automated test suite without relying on external commercial APIs.

**Test Case Weighting System:**
To reflect the varying complexity and criticality of different queries, the final generation accuracy is calculated as a weighted sum. The weights are assigned based on the task difficulty and system safety requirements:
* **Weight 1.0 (Factual)**: Baseline tasks involving direct retrieval and extraction of explicit hardware specifications.
* **Weight 1.5 (Comparison & Ambiguous)**: Advanced tasks requiring multi-chunk synthesis, cross-referencing, or implicit pronoun resolution.
* **Weight 2.0 (Hallucination & Multilingual)**: Critical guardrail tasks. Strict adherence to out-of-domain rejection and cross-lingual constraints is weighted heaviest to ensure the system's reliability and prevent general knowledge bleeding.

### Retrieval Metric (Recall@K)

A query is counted as a Retrieval Hit only when all required retrieval facts defined in `retrieval_facts` are covered by the top-K retrieved chunks.

Retrieval ground truth is evaluated independently from final-answer ground truth to avoid false negatives caused by differences between source-level facts and answer-level expressions.

**Hybrid Judge Scoring Logic:**
The judge verifies the generated response against predefined `answer_facts`. The scoring follows a sequential validation process:
1. **Language & Formatting Guardrails**: Verifies cross-lingual constraints (e.g., no Chinese characters in English responses) and checks for token truncation.
2. **Rejection Verification (Hallucination Defense)**: For queries lacking context or out-of-domain questions, the judge requires exact predefined rejection phrases (e.g., `"規格表中未提供此資訊"` or `"我是專業的技嘉筆電客服..."`).
3. **Alias String Matching**: Normalizes the text (stripping spaces and punctuation) and checks for exact matches against a list of acceptable aliases (e.g., `["2", "兩個", "two"]`).
4. **Semantic Similarity Fallback**: If string matching fails, the judge calculates the cosine similarity between the expected fact and the generated answer using the `multilingual-e5-small` embedding model. A similarity score of $\ge 0.82$ is required to pass.

A test case only scores full points (multiplied by its weight) if it successfully passes all applicable rules. Any missing fact or broken guardrail results in a score of `0`.

## Test result
Run with run_eval.sh which evaluate with 100 test cases generated by GPT-5.6 Luna.

### Test Cases Distribution

| ID Range | Query Type | Weight | Total Cases | Evaluation Focus |
| :--- | :--- | :--- | :--- | :--- |
| **1 - 20** | Factual | 1.0 | 20 | Tests precise retrieval and exact value extraction of hardware specifications. |
| **21 - 40** | Comparison | 1.5 | 20 | Evaluates cross-model specification comparisons and dense information processing. |
| **41 - 60** | Ambiguous | 1.5 | 20 | Tests pronoun resolution and generalized answering when explicit model names are omitted. |
| **61 - 80** | Hallucination Test | 2.0 | 20 | Assesses strict rejection capability for out-of-context information to mitigate over-generation. |
| **81 - 100**| Multilingual | 2.0 | 20 | Tests strict adherence to English-only responses when queried in English against a Traditional Chinese context. |

---
### Benchmark Results

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Generation Accuracy** | **~94.38%** | Overall weighted score by the Hybrid Judge, reflecting factual accuracy and instruction-following capability. |
| **Recall@15** | **100.00%** | All required retrieval facts were covered within the top 15 chunks. |
| **Recall@5** | **98.00%** | 98% of test cases were fully supported by the top 5 retrieved chunks. |

### 💡 Limitations & Case Study (Qwen2.5-3B Error Analysis)

While the system achieves high accuracy, the automated evaluation successfully isolated the physical limitations of running a 3B-parameter model. The five failed test cases can be grouped into four distinct categories of LLM bottlenecks:

#### 1. Attention Slicing in Dense Comparisons (ID: 23)
* **The Error**: When asked to compare the AI Boost frequency of BZH and BYH, the model output `1702 MHz` for both, missing BZH's actual `1797 MHz`.
* **Insight**: When the context is saturated with highly similar numeric hardware specs (e.g., 1797, 1902, 1702), the 3B model's attention mechanism experiences cross-contamination, incorrectly applying attributes of one model to another.

#### 2. Over-Summarization & Omission (ID: 24, 96)
* **The Error**: In ID 24, it correctly listed GPU models, power, and frequencies but omitted the VRAM capacity. In ID 96 (Connectivity Options), it accurately listed 7 different ports but abruptly stopped at Thunderbolt 5, omitting Thunderbolt 4.
* **Insight**: When tasked with generating long, exhaustive lists of technical specifications, the compact model tends to aggressively compress information, prematurely terminating lists (Over-summarization) to save generation space.

#### 3. Deep Semantic Traps (ID: 69)
* **The Error**: The query asked for the screen brightness in *"normal usage mode (SDR)"*. The specification sheet only provides `"500nits (peak)"`. Instead of triggering the rejection protocol, the model confidently answered "500 nits".
* **Insight**: The model acts on keyword reflexes (`Brightness` + `500nits`). It lacks the deeper logical reasoning required to differentiate between "peak" and "normal" states, falling into the semantic trap instead of admitting the information is missing.

#### 4. General Knowledge Bleeding (ID: 75)
* **The Error**: When asked if the RGB keyboard supports per-key customization, the model correctly identified the "3-zone" spec but then hallucinated, stating that per-key details *"might depend on the OS or require checking the user manual,"* instead of strictly rejecting the out-of-context query.
* **Insight**: This is a classic case of parametric memory interference. The model's pre-trained general knowledge about RGB keyboards overrides the strict instruction to answer *only* based on the provided context, leading to speculative generation (hallucination) rather than a clean rejection.

### TTFT, TPS and VRAM test (manual)
The following measurements were collected during interactive inference on the testing environment described above. Values may vary depending on GPU background usage, prompt length, and system load.
| Question | Response | TTFT (s) | TPS | VRAM Usage |
| :--- | :--- | :--- | :--- | :--- |
| What is the GPU of BZH? | The GPU of AORUS MASTER 16 BZH is NVIDIA® GeForce RTX™ 5090 Laptop GPU. | 1.112 | 75.93 | 2369 MB |
| BZH、BYH 和 BXH 的顯示卡有什麼差異？請列出各自的 GPU、VRAM 與最大圖形功耗。 | BZH、BYH 和 BXH 的顯示卡在 GPU、VRAM 和最大圖形功耗方面如下：<br><br>**BZH**:<br>- GPU: NVIDIA® GeForce RTX™ 5090 Laptop GPU<br>- VRAM: 24GB GDDR7<br>- 最大圖形功耗: 175W<br><br>**BYH**:<br>- GPU: NVIDIA® GeForce RTX™ 5080 Laptop GPU<br>- VRAM: 16GB GDDR7<br>- 最大圖形功耗: 175W<br><br>**BXH**:<br>- GPU: NVIDIA® GeForce RTX™ 5070 Ti Laptop GPU<br>- VRAM: 12GB GDDR7<br>- 最大圖形功耗: 140W | 0.799 | 87.33 | 2373 MB |
| BYH 可以選購白色或銀色版本嗎？ | 規格表中未提供此資訊。 | 0.507 | 98.15 | 2373 MB |
| 請幫我寫一段 Python 程式碼 | 我是專業的技嘉筆電客服 AI，只能回答與筆電規格相關的問題。 | 0.641 | 79.42 | 2369 MB |