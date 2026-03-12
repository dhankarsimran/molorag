# MoLoRAG Baseline Reproduction

This folder contains the original code used to reproduce the MoLoRAG paper results. It focus on using large-scale VLM and LLM backends for QA and evaluation.

## Key Scripts

- `main.py`: Main entry point for running QA with various VLMs (QwenVL, DeepSeek-VL, etc.).
- `main_eval.py`: Evaluation script for scoring model outputs.
- `LLMBaseline/`: Contain traditional text-based RAG baselines.
- `VLMRetriever/`: Original retrieval scripts using different methods (base, beamsearch).

## Running Baseline

Execute these commands from the repository root:

- **Run QA Inference**
  ```bash
  python baseline/main.py --dataset MMLong --model_name QwenVL-3B --retriever base
  ```
- **Run Scoring / Evaluation**
  ```bash
  python baseline/main_eval.py --dataset MMLong
  ```
