import json

target_ids = {"gsm8k-test-118-", "gsm8k-test-1043"}

with open("evaluation/scored_candidates.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

rows = raw["rows"]
print(f"rows 的类型是: {type(rows)}，包含 {len(rows)} 条数据")

extracted = []
if isinstance(rows, list):
    for item in rows:
        # 匹配任何包含目标题号的字段
        item_str = json.dumps(item, ensure_ascii=False)
        if any(t in item_str for t in target_ids):
            extracted.append(item)
elif isinstance(rows, dict):
    for k, v in rows.items():
        if any(t in str(k) or t in json.dumps(v, ensure_ascii=False) for t in target_ids):
            extracted.append({"key": k, "content": v})

print(f"成功提取到 {len(extracted)} 个目标条目！")

# 导出为清晰易读的独立文件
with open("two_cases_scored.json", "w", encoding="utf-8") as f:
    json.dump(extracted, f, indent=2, ensure_ascii=False)

print("已保存到 two_cases_scored.json")