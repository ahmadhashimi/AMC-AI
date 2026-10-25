"""Fetch the manifest corpus into JSONL. Intended for a pod with its own disk."""
import argparse
import json
from pathlib import Path
import shutil

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
# Source, config, split, text field, document cap, language tag. Caps keep a pod
# run bounded; docs/CORPUS_MANIFEST.md carries the sizes and licences.
SOURCES = {
    'tinystories_en': ('roneneldan/TinyStories', None, 'train', 'text', 600_000, 'en'),
    'tinystories_fa': ('taesiri/TinyStories-Farsi', None, 'train', None, 152_000, 'fa'),
    'pashto_web': ('HuggingFaceFW/fineweb-2', 'pbt_Arab', 'train', 'text', 400_000, 'ps'),
    'persian_web': ('HuggingFaceFW/fineweb-2', 'fas_Arab', 'train', 'text', 300_000, 'fa'),
    'wikipedia_ps': ('wikimedia/wikipedia', '20231101.ps', 'train', 'text', 0, 'ps'),
    'cosmopedia_stories': ('HuggingFaceTB/cosmopedia', 'stories', 'train', 'text', 200_000, 'en'),
}


def text_of(row, field):
    if field and field in row:
        return row[field]
    for candidate in ('text', 'story', 'content', 'article'):  # schemas differ per dataset
        if candidate in row and isinstance(row[candidate], str):
            return row[candidate]
    raise KeyError(f'No text field in row with keys {list(row)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data/raw/corpus_v1')
    p.add_argument('--only', nargs='*', choices=sorted(SOURCES), help='Fetch a subset')
    p.add_argument('--reserve-gb', type=float, default=5.0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in (args.only or sorted(SOURCES)):
        repo, config, split, field, cap, language = SOURCES[name]
        target = args.out_dir / f'{name}.jsonl'
        if target.exists():
            print(f'{name}: already present, skipping')
            continue
        if shutil.disk_usage(args.out_dir).free < args.reserve_gb * 1024**3:
            raise SystemExit(f'Below the {args.reserve_gb} GiB reserve; stopping before {name}')
        print(f'{name}: streaming {repo} {config or ""} ({"all" if not cap else f"{cap:,}"} docs)', flush=True)
        # Streaming avoids materialising 100 GB datasets on disk for a bounded slice.
        stream = load_dataset(repo, config, split=split, streaming=True)
        partial = target.with_suffix('.partial')
        kept = characters = 0
        with partial.open('w', encoding='utf-8') as sink:
            for row in tqdm(stream, total=cap or None, desc=name, unit='doc'):
                text = text_of(row, field)
                if not text or not text.strip():
                    continue
                sink.write(json.dumps({'text': text, 'language': language, 'source': name},
                                      ensure_ascii=False) + '\n')
                kept += 1
                characters += len(text)
                if cap and kept >= cap:
                    break
        partial.replace(target)
        print(f'{name}: {kept:,} documents, {characters/1e6:.1f}M characters -> {target.name}', flush=True)
    print('Corpus fetch complete. Nothing approved for training yet.')


if __name__ == '__main__':
    main()
