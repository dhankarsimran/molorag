# MoLoRAG+ (Enhanced Multi-Modal RAG)

MoLoRAG+ is an enhanced version of the MoLoRAG retrieval engine, featuring a fine-tuned VLM for superior logical relevance scoring.

## Features

- **Fine-Tuned VLM (LoRA)**: Uses a Qwen2.5-VL-3B model fine-tuned on document logic for more accurate page ranking.
- **Enhanced Retriever**: Integrated logic in `retrieve_plus_v2.py`.
- **Training Scripts**: Includes `train_qwen_lora.py` for further fine-tuning.

## Components

- `molorag_v2_eval.py`: Evaluation script for the Plus version.
- `retrieve_plus_v2.py`: The core retriever logic using the fine-tuned adapter.
- `train_qwen_lora.py`: Training script for LoRA fine-tuning on Qwen2.5-VL.
- `generate_data_qwen.py`: Helper script for generating SFT training data.

## Setup

Requires a fine-tuned adapter path (default: `outputs/final_adapter`).

```bash
python molorag_v2_eval.py
```
