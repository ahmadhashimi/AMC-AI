"""Controlled context-copying test with unseen object/color combinations."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from tqdm import tqdm
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.tokenization import BPETokenizer


def build_examples():
    names = ("احمد", "فرید", "حمید", "بلال")
    objects = ("کتاب", "قلم", "توپ", "بکس")
    colors = ("سرخ", "آبی", "سبز", "زرد")
    places = ("مکتب", "خانه", "پارک")
    splits = {s:[] for s in ("train", "val", "test")}
    for oi, obj in enumerate(objects):
        for ci, color in enumerate(colors):
            split = "val" if ci == oi else "test" if ci == (oi+1)%4 else "train"
            for name in names:
                for place in places:
                    prefix = f"{name} یک {obj} {color} دارد. او به {place} می‌رود. پرسش: {obj} {name} چه رنگ است؟ پاسخ:"
                    splits[split].append({"prompt":prefix,"answer":f"{color}.",
                                          "text":prefix+f" {color}.","object":obj,"color":color})
    return splits


def encode_examples(rows, tokenizer, context):
    encoded = []
    for row in rows:
        prefix = tokenizer.encode(row["prompt"])
        ids = tokenizer.encode(row["text"]) + [tokenizer.eos_id]
        if ids[:len(prefix)] != prefix:
            raise ValueError("Prompt boundary changes tokenization")
        if len(ids)-1 > context:
            raise ValueError("Example exceeds model context")
        encoded.append(ids)
    return encoded


def batch(encoded, indices):
    rows = [encoded[i] for i in indices]
    length = max(len(row)-1 for row in rows)
    x = torch.zeros((len(rows), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, row in enumerate(rows):
        x[i,:len(row)-1] = torch.tensor(row[:-1])
        y[i,:len(row)-1] = torch.tensor(row[1:])
    return x,y


@torch.inference_mode()
def answers(model, tokenizer, rows):
    model.eval()
    seq = [tokenizer.encode(row["prompt"]) for row in rows]
    lengths = [len(s) for s in seq]
    done = [False]*len(rows)
    for _ in range(8):
        x = torch.zeros((len(rows),max(map(len,seq))),dtype=torch.long)
        for i,ids in enumerate(seq): x[i,:len(ids)] = torch.tensor(ids)
        logits,_ = model(x)
        for i,ids in enumerate(seq):
            if done[i]:continue
            token = int(logits[i,len(ids)-1].argmax())
            if token == tokenizer.eos_id:done[i]=True
            else:ids.append(token)
        if all(done):break
    results=[]
    for i,row in enumerate(rows):
        generated = tokenizer.decode(seq[i][lengths[i]:]).strip()
        results.append({"prompt":row["prompt"],"expected":row["answer"],
                        "generated":generated,"exact_match":generated==row["answer"] and done[i]})
    return results


@torch.inference_mode()
def score(model, tokenizer, rows, encoded):
    x,y=batch(encoded,list(range(len(encoded))))
    model.eval()
    _,loss=model(x,y)
    output=answers(model,tokenizer,rows)
    return float(loss),sum(r["exact_match"] for r in output),output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",type=int,default=1500)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--out-dir",type=Path,default=ROOT/"outputs/research/context_seed42")
    p.add_argument("--check-only",action="store_true")
    args=p.parse_args()
    if args.steps<1:p.error("steps must be positive")
    if not args.check_only and args.out_dir.exists():p.error("Output exists; choose a new --out-dir")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    path=ROOT/"artifacts/wikipedia_fa_v1/tokenizer.json"
    tokenizer=BPETokenizer(path)
    cfg=ModelConfig(**json.loads((ROOT/"configs/tiny.json").read_text()))
    cfg.vocab_size=tokenizer.vocab_size
    model=AMCTransformer(cfg)
    splits=build_examples()
    encoded={s:encode_examples(rows,tokenizer,cfg.context_length) for s,rows in splits.items()}
    assert not ({(r['object'],r['color']) for r in splits['train']} & {(r['object'],r['color']) for r in splits['val']})
    print('Fresh model:',model.count_parameters(),'parameters')
    print('Examples:',{s:len(rows) for s,rows in splits.items()})
    print('Chance reference for a balanced four-color guess: 25% (not an exact-generation baseline).')
    print('All words occur in training. Validation object/color pairs do not.')
    print('Test examples remain reserved; this command does not evaluate them.')
    if args.check_only:
        x,y=batch(encoded['train'],[0,1,2,3])
        with torch.inference_mode():_,loss=model(x,y)
        assert torch.isfinite(loss)
        print('PASS: split, tokenization and forward checks. No training or files written.')
        return
    args.out_dir.mkdir(parents=True)
    (args.out_dir/'dataset.json').write_text(json.dumps(splits,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    generator=torch.Generator().manual_seed(args.seed+1)
    report={'experiment':'held_out_object_color_pairs','seed':args.seed,'config':asdict(cfg),
            'tokenizer_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'dataset_sha256':hashlib.sha256(json.dumps(splits,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            'lr':3e-4,'batch_size':4,'measurements':[],
            'limitations':'Template-based context copying only; not broad Dari fluency, reasoning or a pure matched comparison with Wikipedia. All text is original programmatically generated diagnostic material.'}
    def evaluate(step):
        train_loss,train_correct,_=score(model,tokenizer,splits['train'],encoded['train'])
        val_loss,val_correct,output=score(model,tokenizer,splits['val'],encoded['val'])
        row={'step':step,'train_loss':train_loss,'val_loss':val_loss,
             'train_exact':train_correct,'train_total':len(splits['train']),
             'heldout_exact':val_correct,'heldout_total':len(splits['val']),'outputs':output}
        report['measurements'].append(row)
        with (args.out_dir/'progression.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        tqdm.write(f"\nStep {step}: training answers {train_correct}/{len(splits['train'])}; UNSEEN combinations {val_correct}/{len(splits['val'])}; validation loss {val_loss:.4f}")
        for example in output[::12]:
            tqdm.write(f"{example['prompt']}\nExpected: {example['expected']} | Model: {example['generated']}")
    evaluate(0)
    bar=tqdm(range(1,args.steps+1),desc='Context generalization',unit='step')
    try:
        for step in bar:
            indices=torch.randint(len(encoded['train']),(4,),generator=generator).tolist()
            x,y=batch(encoded['train'],indices)
            model.train();optimizer.zero_grad(set_to_none=True)
            _,loss=model(x,y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
            if step%50==0:bar.set_postfix(loss=f'{loss.item():.4f}')
            if step%250==0 or step==args.steps:evaluate(step)
    except KeyboardInterrupt:
        print('Control stopped; measurements preserved. Wikipedia unchanged.')
        return
    (args.out_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    torch.save({'model':model.state_dict(),'config':asdict(cfg),'steps':args.steps,
                'tokenizer_kind':'bpe','tokenizer_path':str(path),'seed':args.seed},args.out_dir/'model.pt')
    print('\nSaved report:',args.out_dir/'report.json')
    print('Held-out exact answer accuracy is a diagnostic result, not a general language-quality score.')


if __name__=='__main__':main()
