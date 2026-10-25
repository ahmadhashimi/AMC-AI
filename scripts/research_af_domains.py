"""Measure how much of FineWeb-2 Persian comes from Afghan sources."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from datasets import load_dataset
from eval_dialect import score as dialect_score

# Afghan publishers that do NOT use a .af domain. Hand-collected; not exhaustive.
AFGHAN_HOSTS = (
    'pajhwok', 'tolonews', '8am.media', 'hasht-e-subh', 'azadiradio', 'avapress',
    'khaama', 'ariananews', 'etilaatroz', 'salaamtimes', 'bakhtarnews', 'wadsam',
    'afghanistantimes', 'afghanpaper', 'jomhornews', 'dw.com/fa-af', 'bbc.com/persian',
    'rawa.org', 'afghan-bios', 'afghanembassy', 'afghanvoice', 'subhekabul',
    'hashtesubh', 'rahapress', 'nn.af', 'ansarpress',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='fas_Arab')
    p.add_argument('--documents', type=int, default=30000)
    p.add_argument('--out', type=Path, default=ROOT / 'docs/research_af_domains.json')
    args = p.parse_args()

    stream = load_dataset('HuggingFaceFW/fineweb-2', args.config, split='train', streaming=True)
    tld = Counter()
    hosts = Counter()
    af_hosts = Counter()
    named_hosts = Counter()
    af_chars = other_chars = 0
    af_samples = []
    seen = 0
    started = time.time()
    for row in stream:
        url = row.get('url') or ''
        text = row.get('text') or ''
        seen += 1
        host = (urlparse(url).hostname or '').lower()
        suffix = host.rsplit('.', 1)[-1] if '.' in host else '?'
        tld[suffix] += 1
        hosts[host] += 1
        is_af = suffix == 'af'
        is_named = any(name in url.lower() for name in AFGHAN_HOSTS)
        if is_af:
            af_hosts[host] += 1
        if is_named:
            named_hosts[host] += 1
        if is_af or is_named:
            af_chars += len(text)
            if len(af_samples) < 12 and len(text) > 500:
                af_samples.append(dict(url=url[:120], chars=len(text), text=text[:300]))
        else:
            other_chars += len(text)
        if seen >= args.documents:
            break

    af_docs = sum(af_hosts.values())
    named_docs = sum(named_hosts.values())
    afghan_docs = af_docs + named_docs
    body = '\n'.join(s['text'] for s in af_samples)
    share, afghan_markers, iranian_markers, _ = dialect_score(body) if body else (0, 0, 0, [])
    report = dict(
        dataset='HuggingFaceFW/fineweb-2', config=args.config, documents_scanned=seen,
        seconds=round(time.time() - started, 1),
        af_tld_documents=af_docs, af_tld_share=round(af_docs / seen, 6),
        named_afghan_host_documents=named_docs,
        afghan_documents_total=afghan_docs, afghan_share=round(afghan_docs / seen, 6),
        afghan_characters=af_chars, other_characters=other_chars,
        afghan_character_share=round(af_chars / (af_chars + other_chars), 6) if af_chars + other_chars else 0,
        top_tlds=dict(tld.most_common(12)),
        af_hosts=dict(af_hosts.most_common(15)),
        named_hosts=dict(named_hosts.most_common(15)),
        dialect_of_afghan_sample=dict(afghan_share=round(share, 4),
                                      afghan_markers=afghan_markers, iranian_markers=iranian_markers,
                                      sample_chars=len(body)),
        projection_full_subset=dict(
            note='Linear projection of the measured share onto the full 58,887,122-document subset. '
                 'The stream is not shuffled, so early documents may not be representative.',
            projected_afghan_documents=round(afghan_docs / seen * 58_887_122),
            projected_afghan_tokens_estimate=round(af_chars / seen * 58_887_122 / 3.3)),
        limitations='Single unshuffled pass over the head of one config. Domain is a proxy for dialect: '
                    'an .af site may publish Iranian Persian and an Afghan author may publish on a .com. '
                    'The named-host list is hand-collected and not exhaustive. No rights review performed.',
        samples=af_samples)
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, ensure_ascii=False, indent=2))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\nwritten:', args.out)


if __name__ == '__main__':
    main()
