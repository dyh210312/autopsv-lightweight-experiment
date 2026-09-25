"""Local batched generation, preserving actual termination tokens and all candidates."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(ROOT/'src'))
from step_segmentation import split_steps_v3

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--questions',type=Path)
    p.add_argument('--model',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=a.questions or ROOT/'data/gsm8k_v3_screened/validation.jsonl'
    data=source.read_bytes()
    raw=json.loads(data) if a.questions else [json.loads(s) for s in data.decode('utf-8').splitlines()]
    groups={r.get('question_id',r.get('id')):r for r in raw}
    if not a.questions and len(groups)!=5:raise ValueError('Expected five fixed development questions')
    modelpath=ROOT.parent/'03_实验程序/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
    modelpath=a.model or modelpath
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    eos=model.generation_config.eos_token_id
    eos_set={eos} if isinstance(eos,int) else set(eos)
    protocol=dict(status='running',source_sha256=hashlib.sha256(data).hexdigest(),
        source=str(source),model_revision=modelpath.name,seed=20260914,max_new_tokens=768,
        candidates_per_question=5,temperature=.7,top_p=.9,
        prompt_suffix='Solve with at most six short reasoning steps. End with a separate line exactly: #### <number>. Do not write anything after that line.',
        scope='development' if not a.questions else 'external frozen questions',eos_ids=sorted(eos_set))
    rows=[];start=time.monotonic()
    def save():
        (a.output/'candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        (a.output/'generation_report.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    save()
    try:
        for qi,(qid,r) in enumerate(sorted(groups.items())):
            set_seed(20260914+qi)
            prompt=tok.apply_chat_template([dict(role='user',content=r['question']+'\n'+protocol['prompt_suffix'])],tokenize=False,add_generation_prompt=True)
            enc=tok(prompt,return_tensors='pt',return_token_type_ids=False).to('cuda')
            with torch.inference_mode():
                generated=model.generate(**enc,num_return_sequences=5,do_sample=True,temperature=.7,top_p=.9,max_new_tokens=768,pad_token_id=tok.pad_token_id)
            for ci,out in enumerate(generated):
                ids=out[enc.input_ids.shape[1]:].tolist()
                stop=next((i for i,t in enumerate(ids) if t in eos_set),None)
                actual=ids if stop is None else ids[:stop+1]
                text=tok.decode(actual,skip_special_tokens=True)
                rows.append(dict(id=f'{qid}-c{ci}',question_id=qid,question=r['question'],
                    gold_answer=r.get('gold_answer',r.get('answer')),solution=text,
                    steps=split_steps_v3(text) if text.strip() else ['[EMPTY RESPONSE]'],
                    finish_reason='length' if stop is None else 'eos',
                    generated_token_ids=actual,generated_tokens=len(actual)))
            protocol['completed_questions']=qi+1;save()
            print(f'Generated {qi+1}/{len(groups)} questions',flush=True)
        protocol['status']='completed'
    except Exception as e:protocol.update(status='failed',error=str(e));raise
    finally:protocol['seconds']=time.monotonic()-start;save()

if __name__=='__main__':main()
