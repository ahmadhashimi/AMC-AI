"""Fetch 100 bounded viewer rows for inspection; never train on truncated previews."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/raw/finepdfs_fa_review'
REPO = 'HuggingFaceFW/finepdfs'


def get(url, limit):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Response exceeds bounded preview limit')
    return json.loads(data)


def main():
    if DEST.exists():
        raise SystemExit(f'Preserving existing sample: {DEST}')
    if shutil.disk_usage(ROOT).free < 1024**3 + 32 * 1024**2:
        raise SystemExit('Need 1 GiB reserve plus a 32 MiB preview budget.')
    info = get(f'https://huggingface.co/api/datasets/{REPO}', 4 * 1024**2)
    revision = info['sha']
    rows = []
    flags = []
    response_revisions = set()
    for offset in tqdm(range(0, 100, 20), desc='Persian sample', unit='batch'):
        query = urllib.parse.urlencode(dict(dataset=REPO, config='fas_Arab',
                                           split='train', offset=offset, length=20))
        data = get('https://datasets-server.huggingface.co/rows?' + query, 4 * 1024**2)
        if data.get('dataset_git_revision'):
            response_revisions.add(data['dataset_git_revision'])
        if len(data['rows']) != 20:
            raise ValueError('Unexpected number of viewer rows')
        for entry in data['rows']:
            row = entry['row']
            text = row.get('text')
            if not isinstance(text, str):
                raise ValueError('Missing text field')
            cues = []
            if entry.get('truncated_cells'):
                cues.append('viewer_truncated_cells')
            if row.get('is_truncated'):
                cues.append('source_truncated')
            if '\ufffd' in text:
                cues.append('replacement_character')
            if any(0xE000 <= ord(c) <= 0xF8FF for c in text):
                cues.append('private_use_glyph')
            if len(text.strip()) < 200:
                cues.append('short_text')
            row_index = entry['row_idx']
            rows.append(dict(row_index=row_index, source=row, review_flags=cues,
                             approved_for_training=False))
            if cues:
                flags.append(dict(row=row_index, flags=cues))
    after = get(f'https://huggingface.co/api/datasets/{REPO}', 4 * 1024**2)['sha']
    if revision != after or any(r != revision for r in response_revisions):
        raise ValueError('Dataset changed or viewer revision differs; refusing mixed snapshot')
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    report = dict(dataset=REPO, config='fas_Arab', split='train', revision=revision,
                  sampling='First 100 viewer rows; not a representative random sample.',
                  viewer_revisions=sorted(response_revisions), rows=len(rows),
                  characters=sum(len(r['source']['text']) for r in rows),
                  sample_sha256=hashlib.sha256(content.encode()).hexdigest(),
                  declared_license='odc-by', flags=flags, approved_for_training=False,
                  limitations='Viewer rows may truncate long cells. No Dari dialect, factual quality or full-document completeness validation. Source metadata preserved.',
                  decision='Inspect first; fetch complete pinned source records before any training if preview cells are truncated.')
    DEST.mkdir(parents=True)
    (DEST / 'sample.jsonl').write_text(content, encoding='utf-8')
    (DEST / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'flags'}, ensure_ascii=False, indent=2))
    print('Flagged rows:', len(flags))
    for row in rows[:3]:
        print('\nROW', row['row_index'], 'URL', row['source'].get('url'))
        print(row['source']['text'][:1200])
    print('Saved review sample only; no training started by this script.')


if __name__ == '__main__':
    main()
