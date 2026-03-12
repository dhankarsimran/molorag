# MoLoRAG Datasets

This folder contains the datasets used for evaluating multi-modal document RAG.

## Structure

- `MMLong/`: PDF documents and page snapshots for the MMLongBench dataset.
- `LongDocURL/`: PDF documents for the LongDocURL dataset.
- `samples_*.json`: Metadata files containing questions, ground truths, and evidence pages.
- `Dataset_Report.md`: A detailed summary of the dataset statistics and information.

## Shared Access

This folder is located at the root to ensure both `molorag_standard` and `molorag_plus` can access the same data files.
