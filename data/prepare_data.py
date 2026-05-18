"""
Prepare instruction-tuning dataset for LLM fine-tuning.
Reads raw text/JSON documents, cleans them, and formats as JSONL.
"""
import json
import re
import argparse
from pathlib import Path


PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Classify and extract key fields from the following document. "
    "Respond ONLY with a valid JSON object.\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,;:!?@#&()\-/\"']", "", text)
    return text.strip()


def format_example(raw: dict) -> dict | None:
    """Convert a raw labeled example into instruction-tuning format."""
    if "text" not in raw or "label" not in raw:
        return None
    cleaned = clean_text(raw["text"])
    if len(cleaned) < 20:
        return None
    output = json.dumps({"label": raw["label"],
                          **raw.get("entities", {})})
    return {
        "instruction": "Classify and extract key fields from the following document. Respond ONLY with a valid JSON object.",
        "input": cleaned[:1500],
        "output": output,
        "text": PROMPT_TEMPLATE.format(input=cleaned[:1500], output=output),
    }


def prepare(input_path: str, output_dir: str, val_ratio: float = 0.1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    examples = [format_example(r) for r in raw_data]
    examples = [e for e in examples if e is not None]

    split = int(len(examples) * (1 - val_ratio))
    train, val = examples[:split], examples[split:]

    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        path = output_dir / name
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"  {name}: {len(data)} examples")

    print(f"Total: {len(examples)} examples → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.val_ratio)
