import json
from pathlib import Path
from .mapper import analyze_catalogue


def run_architecture(root):
    source = root / "reports" / "metadata-latest.json"
    if not source.exists():
        source = root / "docs" / "model-metadata.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    tensors = data.get("tensors", data.get("tensor_catalogue", []))
    result = analyze_catalogue(tensors)
    result["source"] = str(source)
    result["status"] = "PASS" if result["layers"]["verified"] else "REVIEW_REQUIRED"
    out = root / "reports" / "architecture-latest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
