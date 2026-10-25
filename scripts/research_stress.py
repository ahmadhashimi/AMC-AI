"""Challenge the frozen context-control model without additional training."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from amc_model.model import AMCTransformer,ModelConfig
from amc_model.tokenization import BPETokenizer
from research_generalization import build_examples,answers


def conditions():
    tests=build_examples()['test']
    groups={'untouched_test':tests,'changed_color':[],'two_facts_query_first':[],
            'two_facts_query_second':[],'reworded_question':[]}
    colors=['سرخ','آبی','سبز','زرد'];objects=['کتاب','قلم','توپ','بکس']
    for row in tests:
        name=row['prompt'].split(' یک ')[0]
        other='فرید' if name!='فرید' else 'احمد'
        obj=row['object'];color=row['color']
        nextcolor=colors[(colors.index(color)+1)%4]
        nextobj=objects[(objects.index(obj)+1)%4]
        narrative=row['prompt'].split(' پرسش:')[0]
        groups['changed_color'].append({'prompt':row['prompt'].replace(f' {color} دارد.',f' {nextcolor} دارد.'),'answer':nextcolor+'.'})
        facts=narrative+f' {other} یک {nextobj} {nextcolor} دارد.'
        groups['two_facts_query_first'].append({'prompt':facts+f' پرسش: {obj} {name} چه رنگ است؟ پاسخ:','answer':color+'.'})
        groups['two_facts_query_second'].append({'prompt':facts+f' پرسش: {nextobj} {other} چه رنگ است؟ پاسخ:','answer':nextcolor+'.'})
        groups['reworded_question'].append({'prompt':narrative+f' پرسش: رنگ {obj} {name} چیست؟ پاسخ:','answer':color+'.'})
    return groups


def main():
    torch.set_num_threads(2)
    folder=ROOT/'outputs/research/context_seed42'
    target=ROOT/'outputs/research/context_stress_seed42.json'
    if target.exists():raise SystemExit('Stress report exists; preserving it.')
    checkpoint=folder/'model.pt'
    saved=json.loads((folder/'report.json').read_text())
    splits=build_examples()
    assert hashlib.sha256(json.dumps(splits,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==saved['dataset_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    tokenizer_path=Path(state['tokenizer_path'])
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()==saved['tokenizer_sha256']
    tokenizer=BPETokenizer(tokenizer_path)
    model=AMCTransformer(ModelConfig(**state['config'])).eval()
    model.load_state_dict(state['model'])
    report={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'step':state['steps'],'conditions':{},'weights_updated':False,
            'limitations':'One seed, one artificial task, fixed greedy decoding. Test split is now evaluated; do not retune on it and continue to call it untouched. Distractor/paraphrase cases are out of training distribution.'}
    colors=['سرخ','آبی','سبز','زرد']
    for label,rows in conditions().items():
        assert len({r['prompt'] for r in rows})==len(rows)
        assert all(len(tokenizer.encode(r['prompt']))+8<=model.cfg.context_length for r in rows)
        output=answers(model,tokenizer,rows)
        correct=sum(r['exact_match'] for r in output)
        baseline=0
        for row in rows:
            last=max(colors,key=lambda c:row['prompt'].rfind(c))
            baseline+=last+'.'==row['answer']
        report['conditions'][label]={'correct':correct,'total':len(rows),'accuracy':correct/len(rows),
                                     'last_mentioned_color_baseline':baseline/len(rows),'outputs':output}
        print(f'{label}: {correct}/{len(rows)} ({correct/len(rows):.1%}); last-color heuristic {baseline/len(rows):.1%}',flush=True)
        failed=next((r for r in output if not r['exact_match']),None)
        if failed:print('Example failure:',json.dumps(failed,ensure_ascii=False),flush=True)
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print('Saved:',target)
    print('No training performed. Original checkpoint unchanged.')


if __name__=='__main__':main()
