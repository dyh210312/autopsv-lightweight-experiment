import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from common import ROOT, digest, encode, load_base, prefix, read_rows, write_json, write_rows
from answers import extract


def aggregate(scores, method):
    if not scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Invalid probabilities')
    if method == 'final':
        return scores[-1]
    if method == 'min':
        return min(scores)
    if method == 'product':
        return sum(math.log(max(s, 1e-30)) for s in scores)
    raise ValueError('Unknown aggregation: ' + method)


def evaluate(rows, method, k=5):
    groups = defaultdict(list)
    for r in rows:
        groups[r['question_id']].append(r)
    if any(len(g) != k for g in groups.values()):
        raise ValueError(f'Expected exactly {k} candidates per question')
    selected = [max(sorted(g, key=lambda r: r['id']), key=lambda r: aggregate(r['predicted_step_scores'], method)) for g in groups.values()]
    for r in rows:
        if 'label' not in r:
            pred = extract(r.get('solution', ''))
            gold = extract(str(r.get('gold_answer', '')))
            r['label'] = int(pred is not None and gold is not None and pred == gold)
    return dict(aggregation=method, questions=len(groups), candidates_per_question=k,
        top1_rerank_accuracy=sum(r['label'] for r in selected) / len(groups),
        oracle_pass_at_5=sum(any(r['label'] for r in g) for g in groups.values()) / len(groups),
        mean_selected_steps=sum(len(r['steps']) for r in selected) / len(groups),
        selected_ids=[r['id'] for r in selected],
        synthetic=any(r.get('is_synthetic', False) for r in rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data/test.jsonl')
    p.add_argument('--aggregation', choices=['min', 'product', 'final'], default='min')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--output', type=Path, default=ROOT / 'output')
    p.add_argument('--score-only', action='store_true', help='Produce real OSV prefix scores on its training pool')
    p.add_argument('--max-length', type=int, help='Explicit evaluation context limit; never truncate')
    a = p.parse_args()
    report = json.loads((a.adapter / 'training_report.json').read_text(encoding='utf-8'))
    if a.max_length is not None:
        if a.max_length < report['max_length']:
            raise ValueError('Evaluation context override cannot shorten the training context')
        report = dict(report, max_length=a.max_length)
    if Path(a.model).resolve() != Path(report['model']).resolve():
        raise ValueError('Model differs from training model; stage model and adapter together')
    rows = read_rows(a.data)
    if not a.score_only:
        for key, field in [('question_id', 'training_question_ids'), ('question', 'training_questions')]:
            if {r[key] for r in rows} & set(report[field]):
                raise ValueError('Evaluation questions overlap training')
    elif report['target'] != 'outcome':
        raise ValueError('--score-only requires a trained outcome verifier')
    identity = dict(data=digest(a.data), adapter=digest(a.adapter / 'adapter_model.safetensors'),
                    config=digest(a.adapter / 'adapter_config.json'), max_length=report['max_length'],
                    model=str(Path(a.model).resolve()), training_report=digest(a.adapter / 'training_report.json'),
                    device=a.device, scoring_version=2)
    cache = a.output / 'scored_candidates.json'
    cached = json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else None
    if cached and cached['identity'] == identity:
        scored = cached['rows']
    else:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        torch.set_num_threads(4)
        model, tokenizer, dtype = load_base(a.model, a.device, report['quantization'])
        if report['quantization'] == '4bit':
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(model, a.adapter)
        model.eval()
        if 'reload_probe' in report:
            probe = report['reload_probe']
            inputs = tokenizer.pad([encode(tokenizer, probe['text'], report['max_length'])], return_tensors='pt').to(a.device)
            with torch.inference_mode(), torch.autocast(a.device, dtype=dtype, enabled=a.device == 'cuda'):
                actual = model(**inputs).logits.float().item()
            if not math.isclose(actual, probe['logit'], abs_tol=.01, rel_tol=.01):
                raise RuntimeError(f'Saved adapter reload changed score: {actual} != {probe["logit"]}')
            write_json(a.output / 'reload_check.json', dict(status='passed', before=probe['logit'], after=actual,
                       absolute_error=abs(actual - probe['logit']), device=a.device))
        scored = []
        with torch.inference_mode(), torch.autocast(a.device, dtype=dtype, enabled=a.device == 'cuda'):
            for r in rows:
                scores = []
                for start in range(0, len(r['steps']), 4):
                    batch = [encode(tokenizer, prefix(r, i), report['max_length'])
                             for i in range(start, min(start + 4, len(r['steps'])))]
                    inputs = tokenizer.pad(batch, return_tensors='pt').to(a.device)
                    scores.extend(torch.sigmoid(model(**inputs).logits.float()).flatten().cpu().tolist())
                scored.append(dict(r, predicted_step_scores=scores))
        write_json(cache, dict(identity=identity, rows=scored))
    if a.score_only:
        write_rows(a.output / 'osv_scored.jsonl', [dict(r, step_scores=r['predicted_step_scores'],
                    score_source='trained_outcome_adapter:' + identity['adapter']) for r in scored])
    else:
        result = evaluate(scored, a.aggregation)
        write_json(a.output / f'metrics_{a.aggregation}.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
