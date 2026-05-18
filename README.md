# LLM Document Processing Service

> Fine-tuned open-source LLM (Mistral-7B via QLoRA) for domain-specific document classification and entity extraction, deployed as a serverless AWS Lambda inference service backed by S3.

---

## Overview

This project fine-tunes a small open-source large language model for structured information extraction from domain-specific documents. It uses **QLoRA** (Quantized Low-Rank Adaptation) to make fine-tuning feasible on a single GPU, and deploys the model as an **asynchronous serverless service** on AWS Lambda — triggered by S3 uploads and returning structured JSON results.

---

## Architecture

```
         Document Upload
               │
               ▼
      S3 Input Bucket
               │
      S3 Event Trigger
               │
               ▼
    ┌─────────────────────┐
    │   AWS Lambda         │
    │  (inference handler) │
    │                      │
    │  ┌───────────────┐   │
    │  │  QLoRA Model   │   │  ← fine-tuned Mistral-7B adapter
    │  │  (PEFT/HF)    │   │
    │  └───────────────┘   │
    └──────────┬──────────┘
               │
               ▼
      S3 Output Bucket
    (structured JSON results)
```

---

## Features

- **QLoRA Fine-Tuning** — Fine-tunes Mistral-7B with 4-bit quantization using Hugging Face PEFT; GPU memory ~12GB
- **Custom Data Pipeline** — Automated text cleaning, tokenization, and training/validation split generation
- **Structured Output** — Model prompted to return clean JSON (classification label + extracted entities + confidence)
- **Serverless Deployment** — AWS Lambda + S3 trigger for async document processing at scale
- **Benchmarking** — Compares fine-tuned vs base model on accuracy, latency, and inference cost
- **Reproducible Training** — All runs tracked with loss curves and evaluation metrics

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Base Model | Mistral-7B-Instruct-v0.2 |
| Fine-Tuning | Hugging Face PEFT (QLoRA), bitsandbytes, trl |
| Serving | AWS Lambda, S3, boto3 |
| Containerization | Docker (Lambda container image) |
| Experiment Tracking | Weights & Biases (optional) |
| Testing | pytest |
| CI | GitHub Actions |

---

## Project Structure

```
llm-document-processor/
├── data/
│   ├── prepare_data.py        # Clean raw text, build instruction-tuning dataset
│   └── sample_data/
│       └── examples.jsonl     # Example training data format
├── training/
│   ├── finetune.py            # QLoRA fine-tuning script (HF Trainer)
│   ├── config.py              # Training hyperparameters
│   └── utils.py               # Data collator, prompt formatter
├── evaluation/
│   └── benchmark.py           # Compare base vs fine-tuned model on test set
├── inference/
│   ├── handler.py             # AWS Lambda entry point
│   ├── model_loader.py        # Load adapter from S3, initialize pipeline
│   └── postprocess.py         # Parse and validate JSON output
├── deployment/
│   ├── Dockerfile             # Lambda container image
│   └── deploy.sh              # Build, push to ECR, update Lambda
├── tests/
│   ├── test_handler.py
│   └── test_postprocess.py
├── .github/
│   └── workflows/ci.yml
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA GPU (for fine-tuning; ~12GB VRAM for Mistral-7B with QLoRA)
- AWS CLI configured
- Docker (for Lambda deployment)

### Installation

```bash
git clone https://github.com/ahmedsleem109/llm-document-processor.git
cd llm-document-processor
pip install -r requirements.txt
```

### Training Data Format

Each example in `data/` should be a JSONL file:

```json
{"instruction": "Classify and extract key fields from the following document:", "input": "Invoice #12345 from Acme Corp dated 2024-03-01 for $4,500...", "output": "{\"label\": \"invoice\", \"vendor\": \"Acme Corp\", \"amount\": 4500, \"date\": \"2024-03-01\"}"}
```

### Fine-Tune the Model

```bash
python training/finetune.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --data-path data/train.jsonl \
  --output-dir ./outputs/mistral-finetuned \
  --epochs 3 \
  --lora-r 16 \
  --lora-alpha 32
```

### Evaluate

```bash
python evaluation/benchmark.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --finetuned-adapter ./outputs/mistral-finetuned \
  --test-data data/test.jsonl
```

### Deploy to AWS Lambda

```bash
# Build and push Lambda container image to ECR, then update function
bash deployment/deploy.sh
```

### Test the Lambda Locally

```bash
docker build -t llm-doc-processor ./deployment
docker run -p 9000:8080 llm-doc-processor
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"s3_bucket": "my-bucket", "s3_key": "docs/invoice.txt"}'
```

---

## Benchmark Results (Example)

| Metric | Base Model | Fine-Tuned |
|---|---|---|
| Classification Accuracy | 61.2% | 89.7% |
| JSON Parse Success Rate | 43.0% | 96.1% |
| Entity Extraction F1 | 0.54 | 0.88 |
| Avg Inference Latency | 3.8s | 3.6s |

---

## Testing

```bash
pytest tests/ -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
