import json
from transformers import DataCollatorForSeq2Seq


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def compute_json_parse_rate(predictions: list[str]) -> float:
    """Fraction of predictions that parse as valid JSON."""
    success = 0
    for pred in predictions:
        try:
            json.loads(pred.strip())
            success += 1
        except Exception:
            pass
    return success / len(predictions) if predictions else 0.0
