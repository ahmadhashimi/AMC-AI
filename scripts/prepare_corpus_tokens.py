"""Mix the fetched corpus by language share and write uint16 token files."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from amc_model.exclusions import ContaminationError, assert_source_allowed, filter_rows

ROOT = Path(__file__).resolve().parents[1]
# Working mix from docs/CORPUS_MANIFEST.md: Dari-led, English for structure.
MIX = {'fa': 0.50, 'en': 0.25, 'ps': 0.20, 'other': 0.05}


def read_jsonl(path, language=None):
    for line in path.read_text(encoding='utf-8').split('\n'):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get('text', '')
        if text.strip():
            yield row.get('language', language), text


def load_documents(corpus_dir, curriculum):
    """Return {language: [text, ...]}, with sealed evaluation material removed."""
    pools = {}
    for path in sorted(Path(corpus_dir).glob('*.jsonl')):
        for language, text in read_jsonl(path):
            pools.setdefault(language or 'other', []).append(text)
    reports = {}
    if curriculum:
        path = Path(curriculum)
        if not path.exists():
            raise ContaminationError(f'Curriculum source not found: {path}')
        assert_source_allowed(path)          # refuses unfiltered extractions outright
        rows = [json.loads(l) for l in path.read_text(encoding='utf-8').split('\n') if l.strip()]
        kept, report = filter_rows(rows)     # belt and braces: filter even an approved file
        reports['curriculum'] = report
        pools.setdefault('fa', []).extend(r.get('text', '') for r in kept)
        print(f"curriculum: {report['rows_kept']:,} of {report['rows_in']:,} pages kept after "
              f"evaluation exclusion {report['dropped_by_book']}")
    return pools, reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-dir', type=Path, default=ROOT / 'data/raw/corpus_v1')
    p.add_argument('--curriculum', type=Path,
                   default=ROOT / 'data/processed/dari_finetune_v1/train.jsonl',
                   help='Approved Dari corpus. Unfiltered extractions are refused.')
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'artifacts/afghan_v1/tokenizer.json')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data/tokenized/afghan_v1')
    p.add_argument('--target-tokens', type=int, default=500_000_000)
    p.add_argument('--val-fraction', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f'Preserving existing token files: {args.out_dir}')
    # Contamination guard runs before anything else: a safety check that depends
    # on a tokenizer loading first is not a safety check.
    if args.curriculum:
        assert_source_allowed(args.curriculum)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos = tokenizer.token_to_id('<eos>')
    if eos is None or tokenizer.get_vocab_size() > 65535:
        raise SystemExit('Need an <eos> token and a vocabulary that fits uint16')
    pools, exclusion_reports = load_documents(args.corpus_dir, args.curriculum)
    print('pools:', {k: f'{len(v):,} docs' for k, v in pools.items()})

    rng = random.Random(args.seed)
    for documents in pools.values():
        rng.shuffle(documents)

    splits = {'train': [], 'val': []}
    report = {}
    for language, share in MIX.items():
        documents = pools.get(language, [])
        if not documents:
            print(f'{language}: no documents, share redistributed implicitly')
            continue
        budget = int(args.target_tokens * share)
        used = kept = 0
        ids = []
        for text in tqdm(documents, desc=f'{language} ({share:.0%})', unit='doc'):
            encoded = tokenizer.encode(text, add_special_tokens=False).ids
            if not encoded:
                continue
            ids.extend(encoded)
            ids.append(eos)
            used += len(encoded) + 1
            kept += 1
            if used >= budget:
                break
        cut = int(len(ids) * (1 - args.val_fraction))
        splits['train'].append(np.array(ids[:cut], dtype='<u2'))
        splits['val'].append(np.array(ids[cut:], dtype='<u2'))
        report[language] = dict(share=share, documents_used=kept,
                                documents_available=len(documents), tokens=used,
                                budget=budget, budget_met=used >= budget)
        print(f'{language}: {kept:,}/{len(documents):,} docs -> {used:,} tokens '
              f'({"budget met" if used >= budget else "SHORT of budget"})')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'dtype': '<u2', 'bytes_per_token': 2, 'eos_id': eos,
                'tokenizer': str(args.tokenizer.relative_to(ROOT)),
                'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                'mix': MIX, 'per_language': report, 'splits': {},
                'evaluation_exclusions': exclusion_reports}
    for split, parts in splits.items():
        ids = np.concatenate(parts)
        if int(ids[-1]) != eos:   # load_token_files requires the end marker
            ids = np.append(ids, np.array([eos], dtype='<u2'))
        path = args.out_dir / f'{split}.bin'
        ids.tofile(path)
        metadata['splits'][split] = {'tokens': int(ids.size), 'bytes': path.stat().st_size}
        print(f'{split}: {ids.size:,} tokens -> {path.name}')
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
    print('Token files written:', args.out_dir)


if __name__ == '__main__':
    main()
