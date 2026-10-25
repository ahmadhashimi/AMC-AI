"""Read-only regression checks for the saved AMC research models."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from amc_model.model import AMCTransformer,ModelConfig
from amc_model.tokenization import BPETokenizer
from research_sanity import EXAMPLES,greedy
from research_generalization import build_examples,answers


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',action='store_true',help='Write a new baseline manifest; refuses existing manifest')
    p.add_argument('--manifest',type=Path,default=ROOT/'docs/research_baseline_v1.json')
    args=p.parse_args()
    torch.set_num_threads(2)
    tokenizer_path=ROOT/'artifacts/wikipedia_fa_v1/tokenizer.json'
    tokenizer=BPETokenizer(tokenizer_path)
    records={}
    paths={'wikipedia':'outputs/wikipedia_baselines/step_48886.pt',
           'memorization':'outputs/research/sanity_seed42/model.pt',
           'generalization':'outputs/research/context_seed42/model.pt'}
    with torch.inference_mode():
        for name,relative in paths.items():
            path=ROOT/relative
            state=torch.load(path,map_location='cpu',weights_only=True)
            model=AMCTransformer(ModelConfig(**state['config'])).eval()
            model.load_state_dict(state['model'])
            assert model.cfg.vocab_size==tokenizer.vocab_size
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            cfg=model.cfg
            assert cfg.n_layers==3 and cfg.n_heads==4 and cfg.d_model==128 and cfg.context_length==128
            x=torch.tensor([tokenizer.encode('کابل پایتخت')])
            logits,_=model(x)
            repeated,_=model(x)
            torch.testing.assert_close(logits,repeated,rtol=0,atol=0)
            if name=='memorization':
                exact=sum(greedy(model,tokenizer,prompt)==text for prompt,text in EXAMPLES)
                assert exact==4,f'Memorization regression: {exact}/4'
            elif name=='generalization':
                rows=build_examples()['val']
                exact=sum(row['exact_match'] for row in answers(model,tokenizer,rows))
                assert exact==48,f'Generalization regression: {exact}/48'
            else:
                assert state['token_metadata']['tokenizer_sha256']==sha(tokenizer_path)
                exact=None
            records[name]={'path':relative,'sha256':sha(path),'steps':state['steps'],
                           'parameters':model.count_parameters(),'config':state['config'],
                           'known_control_exact_matches':exact}
            print('PASS:',name,'step',state['steps'],'exact matches',exact,flush=True)
    manifest={'tokenizer_sha256':sha(tokenizer_path),'checkpoints':records,
              'environment':{'python':platform.python_version(),**{pkg:importlib.metadata.version(pkg) for pkg in ('torch','numpy','tokenizers')}},
              'scope':'Engineering and narrow control regressions only. No broad language-coherence, medical, voice, vision or MoE quality claim.'}
    if args.record:
        with args.manifest.open('x',encoding='utf-8') as f:
            json.dump(manifest,f,indent=2);f.write('\n')
        print('Recorded baseline:',args.manifest)
    else:
        old=json.loads(args.manifest.read_text())
        assert old['tokenizer_sha256']==manifest['tokenizer_sha256']
        assert old['checkpoints']==manifest['checkpoints'],'Baseline weights/config changed'
        print('PASS: checkpoint and tokenizer hashes match the frozen manifest.')


if __name__=='__main__':main()
