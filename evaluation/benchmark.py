"""
Benchmark base model vs fine-tuned adapter on classification accuracy,
JSON parse rate, and entity extraction F1.
"""
import argparse
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.utils import load_jsonl, compute_json_parse_rate


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def load_model(model_name: str, adapter_path: str | None = None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, examples: list[dict],
                  max_new_tokens: int = 128) -> tuple[list[str], float]:
    predictions = []
    start = time.time()
    for ex in examples:
        prompt = PROMPT_TEMPLATE.format(**ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        predictions.append(generated.strip())
    elapsed = time.time() - start
    return predictions, elapsed


def evaluate_classification(predictions: list[str],
                              ground_truth: list[dict]) -> float:
    correct = 0
    for pred, gt in zip(predictions, ground_truth):
        try:
            parsed = json.loads(pred)
            if parsed.get("label") == gt.get("label"):
                correct += 1
        except Exception:
            pass
    return correct / len(ground_truth) if ground_truth else 0.0


def benchmark(model_name: str, adapter_path: str | None,
              test_path: str, n: int = 50):
    examples = load_jsonl(test_path)[:n]
    ground_truth = [json.loads(ex["output"]) for ex in examples]

    print(f"\nLoading {'fine-tuned' if adapter_path else 'base'} model...")
    model, tokenizer = load_model(model_name, adapter_path)

    preds, elapsed = run_inference(model, tokenizer, examples)
    acc = evaluate_classification(preds, ground_truth)
    parse_rate = compute_json_parse_rate(preds)

    print(f"  Classification accuracy : {acc:.3f}")
    print(f"  JSON parse success rate : {parse_rate:.3f}")
    print(f"  Total inference time    : {elapsed:.1f}s ({elapsed/len(examples)*1000:.0f}ms/doc)")
    return {"accuracy": acc, "json_parse_rate": parse_rate, "latency_ms": elapsed / len(examples) * 1000}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--finetuned-adapter", default=None)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    print("=== BASE MODEL ===")
    base_results = benchmark(args.base_model, None, args.test_data, args.n_samples)

    if args.finetuned_adapter:
        print("\n=== FINE-TUNED MODEL ===")
        ft_results = benchmark(args.base_model, args.finetuned_adapter,
                               args.test_data, args.n_samples)
        print("\n=== COMPARISON ===")
        for k in base_results:
            print(f"  {k}: {base_results[k]:.3f} → {ft_results[k]:.3f}")
