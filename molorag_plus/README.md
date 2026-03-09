# MoLoRAG+: Distilled Logical Retrieval Augmented Generation

MoLoRAG+ is an optimized version of MoLoRAG that uses knowledge distillation to train a smaller, local retriever.

## 1. Dependencies
- Python 3.10+
- PyTorch (with MPS support for Mac)
- Transformers & PEFT (HuggingFace)
- qwen_vl_utils
- fitz (PyMuPDF)

Install:
```bash
pip install torch transformers peft qwen_vl_utils pymupdf
```

## 2. Data Generation (Distillation)
To generate the training triplets (Question, Image, Score) using a Qwen teacher model:
```bash
python generate_data_qwen.py
```
This script samples document pages and uses the teacher model to assign logical relevance scores.

## 3. Training (LoRA Fine-tuning)
To fine-tune the Qwen2.5-VL-3B model on your local MacBook:
```bash
python train_qwen_lora.py
```
Key parameters are optimized for Apple Silicon (MPS).

## 4. Evaluation
To evaluate the fine-tuned MoLoRAG+ v2 model:
```bash
python molorag_v2_eval.py
```
This script tests the model on MMLongBench and LongDocURL datasets.

## 5. Pretrained Models
- **Teacher**: Qwen2.5-VL-3B-Instruct (Local distillation)
- **Student**: Qwen2.5-VL-3B (Fine-tuned with LoRA)
