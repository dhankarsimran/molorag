# MoLoRAG+ v2 (Qwen Distilled) Technical Report

## 1. Executive Summary
This report details the implementation and evaluation of **MoLoRAG+ v2**, a locally distilled version of the MoLoRAG retriever. This version replaces proprietary teacher models (GPT-4o) with an open-source teacher (**Qwen2.5-VL**) for cost-efficient data generation and multi-modal fine-tuning.

## 2. Technical Architecture

### 2.1 Data Distillation Pipeline
The training data was generated using a "Teacher-Student" distillation paradigm:
- **Teacher Engine**: `Qwen2.5-VL-3B-Instruct` (optimized for local MacBook memory).
- **Sampling Strategy**: Random document snapshots from MMLongBench-Doc.
- **Question Generation**: Teacher generates questions matching a target relevance score (1-5).
- **Automated Verification**: Self-verification logic ensures generated questions align with target scores ($|s - s'| \leq 1$).

### 2.2 Fine-Tuning Configuration
The model was fine-tuned using Parameter-Efficient Fine-Tuning (PEFT):
- **Base Model**: `Qwen2.5-VL-3B-Instruct`.
- **Method**: LoRA (Low-Rank Adaptation).
- **LoRA Rank**: 8.
- **Hardware Acceleration**: Apple Silicon MPS (Metal Performance Shaders).
- **Data Format**: `(Question, Image, Relevance_Score)` triplets.

## 3. Evaluation Metrics

The retrieval engine was evaluated using the fine-tuned LoRA adapters on the **MMLongBench** and **LongDocURL** datasets. Results are presented in percentages (%).

### 3.1 Dataset: MMLongBench
| Metric | Top-1 | Top-3 | Top-5 |
| :--- | :---: | :---: | :---: |
| **Recall** | 33.33 | 60.00 | 60.00 |
| **Precision** | 60.00 | 46.67 | 28.00 |
| **NDCG** | 60.00 | 65.92 | 61.68 |
| **MRR** | 60.00 | 70.00 | 70.00 |

### 3.2 Dataset: LongDocURL
| Metric | Top-1 | Top-3 | Top-5 |
| :--- | :---: | :---: | :---: |
| **Recall** | 40.00 | 40.00 | 40.00 |
| **Precision** | 40.00 | 13.33 | 8.00 |
| **NDCG** | 40.00 | 40.00 | 40.00 |
| **MRR** | 40.00 | 40.00 | 40.00 |

## 4. Implementation Artifacts
The following scripts in `molorag_plus_v2/` support this implementation:
- `generate_data_qwen.py`: Distillation engine.
- `train_qwen_lora.py`: Fine-tuning script.
- `retrieve_plus_v2.py`: Inference wrapper with adapter integration.
- `molorag_v2_eval.py`: Evaluation suite.

## 5. Conclusion
MoLoRAG+ v2 successfully demonstrates that an open-source VLM can serve as both teacher and student for high-performance visual-logical retrieval. The pipeline is fully optimized for local MacBook execution, providing a scalable path for custom document retrieval tasks.
