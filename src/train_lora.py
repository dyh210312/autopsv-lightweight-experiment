"""Prefix verifier with explicit BCE or probability-MSE objective."""
import argparse
import time
from pathlib import Path
from common import ROOT, digest, encode, load_base, prefix, read_rows, write_json, supervised_steps
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed


class BCETrainer(Trainer):
    loss_kind = 'bce'

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels').float()
        outputs = model(**inputs)
        logits = outputs.logits.float().view(-1)
        loss = (torch.nn.functional.mse_loss(logits.sigmoid(), labels.view(-1))
                if self.loss_kind == 'mse' else
                torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.view(-1)))
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss')
        return (loss, outputs) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Pre-downloaded local model directory')
    p.add_argument('--data', type=Path, default=ROOT / 'data/annotated_theta_0.5.jsonl')
    p.add_argument('--validation', type=Path, default=ROOT / 'data/validation.jsonl')
    p.add_argument('--output', type=Path, default=ROOT / 'output/adapter')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--quantization', choices=['none', '4bit'], default='none')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--target', choices=['process', 'outcome'], default='process')
    p.add_argument('--outcome-scope', choices=['prefixes','final'], default='prefixes',
                   help='final supervises only the complete solution; explicit OSV ablation')
    p.add_argument('--init-adapter', help='Continue a trained OSV adapter for process training')
    p.add_argument('--resume', help='Trainer checkpoint directory, including optimizer state')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--loss', choices=['bce', 'mse'], default='bce',
                   help='Explicit objective; mse operates on sigmoid probabilities')
    a = p.parse_args()
    if a.target != 'outcome' and a.outcome_scope != 'prefixes':
        raise ValueError('--outcome-scope final requires --target outcome')
    if a.resume and a.loss != 'bce':
        raise ValueError('MSE trial must start fresh; objective-changing resume is disabled')
    if (a.output / 'training_report.json').exists() and not a.resume:
        raise FileExistsError('Use a fresh output directory, or explicitly --resume')
    set_seed(a.seed)
    torch.set_num_threads(4)
    start = time.monotonic()
    rows, validation = read_rows(a.data), read_rows(a.validation)
    for key in ('question_id', 'question'):
        if {r[key] for r in rows} & {r[key] for r in validation}:
            raise ValueError('Train/validation question leakage')
    model, tokenizer, dtype = load_base(a.model, a.device, a.quantization)
    if a.quantization == '4bit':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=['q_proj', 'v_proj'],
            r=8, lora_alpha=16, lora_dropout=.05, modules_to_save=['score'], bias='none'))
    examples = []
    for r in rows:
        for i, y in supervised_steps(r, a.target, a.outcome_scope):
            examples.append(dict(encode(tokenizer, prefix(r, i), a.max_length), labels=float(y)))
    if not examples:
        raise ValueError('No observed supervision remains after masking')
    trainable, total = model.get_nb_trainable_parameters()
    initial = {n: v.detach().cpu().clone() for n, v in model.named_parameters() if v.requires_grad}
    args = TrainingArguments(output_dir=str(a.output), per_device_train_batch_size=4,
        gradient_accumulation_steps=2, max_steps=30, learning_rate=2e-4,
        logging_steps=5, save_steps=15, save_total_limit=2, report_to='none',
        bf16=a.device == 'cuda' and dtype == torch.bfloat16,
        fp16=a.device == 'cuda' and dtype == torch.float16, use_cpu=a.device == 'cpu',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=0, dataloader_pin_memory=a.device == 'cuda', seed=a.seed,
        optim='adamw_torch', remove_unused_columns=False, disable_tqdm=True)
    trainer = BCETrainer(model=model, args=args, train_dataset=examples,
                         data_collator=DataCollatorWithPadding(tokenizer), tokenizer=tokenizer)
    trainer.loss_kind = a.loss
    trainer.train(resume_from_checkpoint=a.resume)
    if trainer.state.global_step != 30:
        raise RuntimeError('Incomplete optimization: expected exactly 30 updates')
    changed = [n for n, v in model.named_parameters() if n in initial and not torch.equal(initial[n], v.detach().cpu())]
    if not a.resume and (not any('lora_B' in n for n in changed) or not any('score' in n for n in changed)):
        raise RuntimeError('LoRA/head did not both update')
    model.eval()
    probe_text = prefix(rows[0], len(rows[0]['steps']) - 1)
    probe_inputs = tokenizer.pad([encode(tokenizer, probe_text, a.max_length)], return_tensors='pt').to(a.device)
    with torch.inference_mode():
        probe_logit = model(**probe_inputs).logits.float().item()
    trainer.save_model(str(a.output))
    tokenizer.save_pretrained(a.output)
    reserved = torch.cuda.max_memory_reserved() if a.device == 'cuda' else 0
    report = dict(status='completed', steps=trainer.state.global_step, trajectories=len(rows), prefix_examples=len(examples),
        masked_prefixes=sum(len(r['steps']) for r in rows)-len(examples),
        trainable_parameters=trainable, total_parameters=total, trainable_percent=100 * trainable / total,
        changed_trainable_tensors=len(changed), seconds=time.monotonic() - start, peak_reserved_bytes=reserved,
        device=a.device, dtype=str(dtype), quantization=a.quantization, model=str(Path(a.model).resolve()),
        data_sha256=digest(a.data), training_question_ids=sorted({r['question_id'] for r in rows}),
        training_questions=sorted({r['question'] for r in rows}), target=a.target,
        outcome_scope=a.outcome_scope if a.target=='outcome' else None,
        max_length=a.max_length, seed=a.seed, synthetic=any(r.get('is_synthetic', False) for r in rows),
        reload_probe=dict(text=probe_text, logit=probe_logit),
        loss=('MSE on sigmoid confidence; SeqCls/LoRA variant' if a.loss == 'mse' else
              'BCEWithLogits, sigmoid confidence; differs from original paper MSE'),
        torch=torch.__version__)
    write_json(a.output / 'training_report.json', report)
    model.print_trainable_parameters()
    print({k: report[k] for k in ('status', 'steps', 'trajectories', 'prefix_examples', 'seconds', 'peak_reserved_bytes', 'synthetic')})


if __name__ == '__main__':
    main()
