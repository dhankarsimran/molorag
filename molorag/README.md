# MoLoRAG: Multi-modal Logical Retrieval Augmented Generation (Standard Version)

This folder contains the reproduction code for the standard MoLoRAG system as described in the original paper.

## 1. Dependencies
- Python 3.10+
- PyTorch 2.4+
- Transformers
- CLIP (OpenAI)
- PyMuPDF (fitz)
- NetworkX
- Pillow

Install dependencies:
```bash
pip install torch transformers networkx pymupdf pillow
```

## 2. Data Download Instructions
The dataset used is **MMLongBench-Doc**.
1. Download the PDFs and question-answer pairs from the [official MoLoRAG repository](https://github.com/WxxShirley/MoLoRAG).
2. Place the PDFs in the `dataset/` folder and the JSONL files in the `data/` folder.

## 3. Preprocessing
To generate embeddings for the document pages:
```bash
python main.py --mode index --data_path dataset/mmlongbench
```

## 4. Evaluation
To run the MoLoRAG retrieval engine:
```bash
python molorag_local_eval.py
```
This script implements Algorithm 1 (Hierarchical Traversal) and calculates Recall, Precision, NDCG, and MRR.

## 5. Pretrained Models
- **Visual Encoder**: `openai/clip-vit-large-patch14`
- **Logical Reasoner**: `Qwen/Qwen2.5-VL-3B-Instruct`
