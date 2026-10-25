"""Balanced one-/two-fact training; evaluate the previously exposed stress set."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from tqdm import tqdm
from amc_model.model import AMCTransformer,ModelConfig
from amc_model.tokenization import BPETokenizer
from research_generalization import build_examples,encode_examples,batch,answers
from research_stress import conditions


def training_examples():
    original=build_examples()['train']
    pairs=sorted({(r['object'],r['color']) for r in original})
    rows=[dict(r) for r in original]
    names=['احمد','فرید','حمید','بلال']
    for first in original:
        name=first['prompt'].split(' یک ')[0]
        other=names[(names.index(name)+1)%len(names)]
        narrative=first['prompt'].split(' پرسش:')[0]
        for obj,color in pairs:
            if obj==first['object'] or color==first['color']:continue
            facts=narrative+f' {other} یک {obj} {color} دارد.'
            for query_obj,query_name,answer in ((first['object'],name,first['color']),(obj,other,color)):
                prompt=facts+f' پرسش: {query_obj} {query_name} چه رنگ است؟ پاسخ:'
                rows.append({'prompt':prompt,'answer':answer+'.','text':prompt+' '+answer+'.',
                             'fact_pairs':[[first['object'],first['color']],[obj,color]]})
    return list({r['text']:r for r in rows}.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--out-dir',type=Path,default=ROOT/'outputs/research/binding_seed42')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.steps<1:p.error('steps must be positive')
    if not args.check_only and args.out_dir.exists():p.error('Output exists; preserving it')
    torch.set_num_threads(2);torch.manual_seed(42)
    token_path=ROOT/'artifacts/wikipedia_fa_v1/tokenizer.json'
    t=BPETokenizer(token_path)
    cfg=ModelConfig(**json.loads((ROOT/'configs/tiny.json').read_text()));cfg.vocab_size=t.vocab_size
    model=AMCTransformer(cfg)
    rows=training_examples();encoded=encode_examples(rows,t,cfg.context_length)
    train_pairs={(r['object'],r['color']) for r in build_examples()['train']}
    for r in rows:
        assert all(tuple(pair) in train_pairs for pair in r.get('fact_pairs',[]))
    groups=conditions()
    print('Fresh model; balanced training examples:',len(rows),flush=True)
    print('Stress cases are development evaluations, not a new untouched test.',flush=True)
    if args.check_only:
        x,y=batch(encoded,[0,1,2,3])
        with torch.inference_mode():_,loss=model(x,y)
        assert torch.isfinite(loss)
        print('PASS: data/pair exclusion and forward checks; no training.')
        return
    args.out_dir.mkdir(parents=True)
    serialized=json.dumps(rows,ensure_ascii=False,sort_keys=True)
    (args.out_dir/'training_data.json').write_text(serialized,encoding='utf-8')
    report={'config':asdict(cfg),'seed':42,'steps':args.steps,'training_examples':len(rows),
            'training_sha256':hashlib.sha256(serialized.encode()).hexdigest(),
            'tokenizer_sha256':hashlib.sha256(token_path.read_bytes()).hexdigest(),
            'lr':3e-4,'batch_size':4,'progression':[],
            'limitations':'Adaptive development experiment: stress cases informed data design. One seed, different data volume/token budget than the first control. Not proof of architecture causality or broad coherence.'}
    def evaluate(step):
        row={'step':step,'conditions':{}}
        for name,examples in groups.items():
            result=answers(model,t,examples)
            score=sum(r['exact_match'] for r in result)
            row['conditions'][name]={'correct':score,'total':len(examples),'outputs':result}
        report['progression'].append(row)
        with (args.out_dir/'progression.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        tqdm.write('Step '+str(step)+': '+', '.join(f'{k} {v["correct"]}/{v["total"]}' for k,v in row['conditions'].items()))
    evaluate(0)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    rng=torch.Generator().manual_seed(43)
    bar=tqdm(range(1,args.steps+1),desc='Balanced object binding',unit='step')
    for step in bar:
        model.train()
        x,y=batch(encoded,torch.randint(len(encoded),(4,),generator=rng).tolist())
        opt.zero_grad(set_to_none=True);_,loss=model(x,y)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
        if step%100==0:bar.set_postfix(loss=f'{loss.item():.4f}')
        if step%500==0 or step==args.steps:evaluate(step)
    torch.save({'model':model.state_dict(),'config':asdict(cfg),'steps':args.steps,
                'tokenizer_kind':'bpe','tokenizer_path':str(token_path)},args.out_dir/'model.pt')
    (args.out_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved:',args.out_dir/'report.json')


if __name__=='__main__':main()
