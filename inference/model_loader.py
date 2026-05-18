"""
Load fine-tuned model adapter from S3 and initialize inference pipeline.
Cached after first load (Lambda warm start optimization).
"""
import os
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel


def download_adapter_from_s3(s3_uri: str, local_dir: str = "/tmp/adapter") -> str:
    """Download LoRA adapter from S3 to /tmp."""
    if not s3_uri:
        return None
    s3 = boto3.client("s3")
    s3_uri = s3_uri.replace("s3://", "")
    bucket, prefix = s3_uri.split("/", 1)
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)

    return local_dir


def load_pipeline(adapter_s3_path: str | None = None,
                  model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Load model and return HuggingFace text-generation pipeline."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto")

    if adapter_s3_path:
        local_adapter = download_adapter_from_s3(adapter_s3_path)
        model = PeftModel.from_pretrained(model, local_adapter)

    model.eval()
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
