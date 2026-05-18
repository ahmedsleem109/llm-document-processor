"""
Parse and validate JSON output from the LLM.
Handles common generation artifacts like markdown code fences.
"""
import json
import re


REQUIRED_FIELDS = {"label"}


def parse_output(raw: str) -> dict:
    """
    Extract structured JSON from LLM output.
    Handles markdown fences, leading text, and trailing content.
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()

    # Try direct parse
    try:
        result = json.loads(raw)
        return _validate(result)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object from raw text
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return _validate(result)
        except json.JSONDecodeError:
            pass

    return {"label": "unknown", "raw_output": raw, "parse_error": True}


def _validate(result: dict) -> dict:
    """Ensure required fields are present."""
    if not isinstance(result, dict):
        return {"label": "unknown", "parse_error": True}
    for field in REQUIRED_FIELDS:
        if field not in result:
            result[field] = "unknown"
    return result
