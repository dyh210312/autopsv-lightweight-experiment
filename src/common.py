"""Shared validation and model loading; never silently truncate or invent labels."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def read_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows:
        raise ValueError('Empty dataset')
    seen = set()
    questions = {}
    for r in rows:
        if r['id'] in seen:
            raise ValueError('Duplicate trajectory id: ' + r['id'])
        seen.add(r['id'])
        if not isinstance(r['question'], str) or not r['question'].strip():
            raise ValueError('Missing question')
        if questions.setdefault(r['question_id'], r['question']) != r['question']:
            raise ValueError('Question id maps to different questions')
        if not r['steps'] or any(not isinstance(s, str) or not s.strip() for s in r['steps']):
            raise ValueError('Empty reasoning steps')
        if type(r['label']) not in (int, bool) or r['label'] not in (0, 1):
            raise ValueError('Outcome labels must be verified 0/1, not unknown answers')
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prefix(r, step):
    return 'Question: ' + r['question'] + '\nSolution so far:\n' + '\n'.join(r['steps'][:step + 1])


def deltas(scores, formula):
    if len(scores) < 1 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('step_scores must be finite probabilities in [0,1]')
    if formula == 'relative' and any(s <= 1e-8 for s in scores[:-1]):
        raise ValueError('Relative change is undefined/unstable near zero; inspect OSV scores')
    return [(b - a) / a if formula == 'relative' else b - a for a, b in zip(scores, scores[1:])]


def annotate(scores, theta, formula):
    # No question-only P0 is supplied. First step is unobserved, assigned 1 by convention.
    labels, failed = [1], False
    for d in deltas(scores, formula):
        failed = failed or d <= theta
        labels.append(int(not failed))
    return labels


def encode(tokenizer, text, max_length):
    encoded = tokenizer(text, truncation=False, return_token_type_ids=False)
    if len(encoded['input_ids']) > max_length:
        raise ValueError(f'Prefix has {len(encoded["input_ids"])} tokens > max_length={max_length}; refusing silent truncation')
    return encoded


def supervised_steps(row, target, outcome_scope='prefixes'):
    """None means unobserved; only the process first step may be masked."""
    if outcome_scope not in ('prefixes', 'final'):
        raise ValueError('Invalid outcome scope')
    if target != 'outcome' and outcome_scope != 'prefixes':
        raise ValueError('Outcome scope only applies to outcome supervision')
    if target == 'outcome' and outcome_scope == 'final':
        if not row['steps'] or type(row['label']) not in (int, bool) or row['label'] not in (0,1):
            raise ValueError('Invalid final outcome supervision')
        return [(len(row['steps'])-1,row['label'])]
    labels = [row['label']] * len(row['steps']) if target == 'outcome' else row['step_labels']
    if len(labels) != len(row['steps']):
        raise ValueError('Step/label length mismatch')
    result = []
    for i, label in enumerate(labels):
        if label is None and i == 0 and target == 'process':
            continue
        if type(label) not in (int, bool) or label not in (0, 1):
            raise ValueError('Invalid step label; only unobserved first step may be None')
        result.append((i, label))
    return result


def load_base(model_path, device, quantization='none', memory_gib=22):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(memory_gib * 2**30 / total, .95), 0)
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == 'cuda' else torch.float32
    kwargs = dict(num_labels=1, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    if quantization == '4bit':
        if device != 'cuda':
            raise ValueError('4-bit profile requires CUDA and bitsandbytes')
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype), device_map={'': 0})
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer needs an existing padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    if not hasattr(model, 'score'):
        raise ValueError('This profile requires a score classification head (Mistral/Qwen2)')
    # Transformers 4.45 can leave a NEW SeqCls head on meta when loading causal-LM
    # weights with low_cpu_mem_usage. Initialize only this missing, trainable head.
    if model.score.weight.is_meta:
        model.score.to_empty(device='cpu' if quantization == 'none' else device)
        model._init_weights(model.score)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError('Backbone still contains unloaded meta tensors; verify model weights')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if quantization == 'none':
        model.to(device)
    return model, tokenizer, dtype
