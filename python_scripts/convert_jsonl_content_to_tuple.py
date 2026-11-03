import json

def convert_jsonl_content_to_tuple(path):
    with open(path, encoding="utf-8") as f:
        return [(r["file_name"], r["text"])
                for r in map(json.loads, f) if r["text"].strip()]