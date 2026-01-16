# GraphRAG with SAT Reasoning

This repository implements a Graph-based Retrieval-Augmented Generation (GraphRAG)
pipeline with Structure-Aware Tuning (SAT).

## Features
- Semantic + graph hybrid retrieval
- Knowledge Graph construction with spaCy
- Local SAT reasoning (Qwen + LoRA)
- Optional remote LLM generation (Kimi K2)

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

nhớ import API key: 
$env:NVAPI_KEY=""
