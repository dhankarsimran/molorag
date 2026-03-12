# MoLoRAG Standard Evaluation

This module provides the standard, self-contained evaluation engine for the MoLoRAG retrieval system. It implements the hierarchical logic-aware traversal algorithm designed for high-performance multi-modal document retrieval.

## Overview

The standard implementation uses:
- **CLIP (openai/clip-vit-large-patch14)**: For semantic page embeddings and initial proximity graph construction.
- **Qwen2.5-VL-3B-Instruct**: As the Visual Language Model (VLM) for logical relevance scoring.

## Key Components

- `molorag_local_eval.py`: The main execution script. It processes PDF documents, builds a semantic graph, and performs a multi-hop traversal to find the most relevant pages for a given query.

## How to Run

Ensure you have the `dataset/` folder at the root with the required JSON samples and PDF documents.

```bash
# Run evaluation (execute from the repository root)
python molorag/molorag_standard/molorag_local_eval.py
```

## Features
- **MacBook/MPS Optimization**: Automatically uses Apple Silicon GPU for faster inference.
- **Hierarchical Traversal**: Combines semantic similarity with deep visual understanding.
- **Metrics**: Calculates Recall, Precision, NDCG, and MRR.
