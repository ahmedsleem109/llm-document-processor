"""
AWS Lambda entry point for document processing inference.
Triggered by S3 upload events. Downloads the document, runs inference,
and writes structured JSON results back to the output S3 bucket.
"""
import json
import os
import boto3
from inference.model_loader import load_pipeline
from inference.postprocess import parse_output

s3 = boto3.client("s3")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "my-output-bucket")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        adapter_path = os.environ.get("ADAPTER_S3_PATH", "")
        _pipeline = load_pipeline(adapter_path)
    return _pipeline


PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Classify and extract key fields from the following document. "
    "Respond ONLY with a valid JSON object.\n\n"
    "### Input:\n{text}\n\n"
    "### Response:\n"
)


def handler(event, context):
    """Lambda handler — processes one document per invocation."""
    bucket = event["s3_bucket"]
    key = event["s3_key"]

    response = s3.get_object(Bucket=bucket, Key=key)
    document_text = response["Body"].read().decode("utf-8")

    prompt = PROMPT_TEMPLATE.format(text=document_text[:2000])
    pipe = get_pipeline()
    raw_output = pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    raw_output = raw_output[len(prompt):]

    result = parse_output(raw_output)
    result["source_key"] = key

    output_key = key.replace("input/", "output/").replace(".txt", ".json")
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=output_key,
        Body=json.dumps(result, indent=2),
        ContentType="application/json",
    )

    return {"statusCode": 200, "output_key": output_key, "result": result}
