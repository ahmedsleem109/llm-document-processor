import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import pytest
from inference.postprocess import parse_output
from training.utils import load_jsonl, compute_json_parse_rate


# ── Postprocessing ─────────────────────────────────────────────────────────────
class TestPostprocess:
    def test_parses_clean_json(self):
        raw = '{"label": "invoice", "vendor": "Acme", "amount": 4500}'
        result = parse_output(raw)
        assert result["label"] == "invoice"
        assert result["vendor"] == "Acme"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"label": "contract"}\n```'
        result = parse_output(raw)
        assert result["label"] == "contract"

    def test_extracts_json_from_surrounding_text(self):
        raw = 'Here is the result: {"label": "receipt", "total": 31.34} Hope that helps!'
        result = parse_output(raw)
        assert result["label"] == "receipt"

    def test_returns_unknown_on_unparseable(self):
        result = parse_output("Sorry, I cannot process this document.")
        assert result["label"] == "unknown"
        assert result.get("parse_error") is True

    def test_adds_missing_label_field(self):
        raw = '{"vendor": "Acme", "amount": 500}'
        result = parse_output(raw)
        assert "label" in result

    def test_handles_empty_string(self):
        result = parse_output("")
        assert result["label"] == "unknown"

    def test_handles_nested_json(self):
        raw = '{"label": "invoice", "entities": {"vendor": "Corp", "items": [1, 2]}}'
        result = parse_output(raw)
        assert result["label"] == "invoice"
        assert "entities" in result


# ── Training Utils ─────────────────────────────────────────────────────────────
class TestTrainingUtils:
    def test_load_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        data = [{"instruction": "do x", "input": "doc", "output": '{"label":"a"}'}]
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
        loaded = load_jsonl(str(path))
        assert len(loaded) == 1
        assert loaded[0]["instruction"] == "do x"

    def test_load_jsonl_skips_empty_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"label": "a"}\n\n{"label": "b"}\n')
        loaded = load_jsonl(str(path))
        assert len(loaded) == 2

    def test_json_parse_rate_all_valid(self):
        preds = ['{"label": "a"}', '{"label": "b"}', '{"label": "c"}']
        assert compute_json_parse_rate(preds) == 1.0

    def test_json_parse_rate_none_valid(self):
        preds = ["not json", "also not json", "hello world"]
        assert compute_json_parse_rate(preds) == 0.0

    def test_json_parse_rate_partial(self):
        preds = ['{"label": "a"}', "invalid", '{"label": "b"}']
        rate = compute_json_parse_rate(preds)
        assert abs(rate - 2/3) < 0.01

    def test_json_parse_rate_empty(self):
        assert compute_json_parse_rate([]) == 0.0


# ── Data Preparation ───────────────────────────────────────────────────────────
class TestDataPreparation:
    def test_prepare_creates_splits(self, tmp_path):
        from data.prepare_data import prepare
        input_file = tmp_path / "raw.jsonl"
        examples = [
            {"text": f"Document number {i} with enough content to pass filter.",
             "label": "invoice", "entities": {"amount": i * 100}}
            for i in range(20)
        ]
        with open(input_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        out_dir = str(tmp_path / "output")
        prepare(str(input_file), out_dir, val_ratio=0.2)

        train_path = os.path.join(out_dir, "train.jsonl")
        val_path = os.path.join(out_dir, "val.jsonl")
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)

        train_data = load_jsonl(train_path)
        val_data = load_jsonl(val_path)
        assert len(train_data) + len(val_data) == 20
        assert len(val_data) == 4   # 20% of 20

    def test_format_example_structure(self):
        from data.prepare_data import format_example
        raw = {"text": "Invoice from Acme Corp for $500 dated today with full details.",
               "label": "invoice", "entities": {"vendor": "Acme"}}
        result = format_example(raw)
        assert result is not None
        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert "text" in result

    def test_format_example_rejects_short_text(self):
        from data.prepare_data import format_example
        result = format_example({"text": "short", "label": "x"})
        assert result is None

    def test_format_example_rejects_missing_fields(self):
        from data.prepare_data import format_example
        result = format_example({"text": "some text without label field"})
        assert result is None
