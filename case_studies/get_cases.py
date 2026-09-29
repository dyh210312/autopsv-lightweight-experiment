"""Extract the two report case studies from the scored candidate data."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "evaluation" / "scored_candidates.json"
OUTPUT = Path(__file__).with_name("two_cases_scored.json")
TARGET_IDS = {"gsm8k-test-118", "gsm8k-test-1043"}

with SOURCE.open("r", encoding="utf-8") as file:
    rows = json.load(file)["rows"]

extracted = [row for row in rows if row.get("question_id") in TARGET_IDS]

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(extracted, file, indent=2, ensure_ascii=False)

print(f"Saved {len(extracted)} candidate records to {OUTPUT}")
