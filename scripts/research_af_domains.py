"""Measure how much of FineWeb-2 Persian comes from Afghan sources."""
import argparse
from collections import Counter
import hashlib
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

# Afghan publishers that do not use a .af domain. Matched on the HOSTNAME with a
# label boundary, never as a substring of the whole URL: "nn.af" inside a path or
# "bbc.com" anywhere would otherwise produce false positives.
AFGHAN_HOSTS = (
    'pajhwok.com', 'tolonews.com', '8am.media', 'hasht-e-subh.com', 'hashtesubh.com',
    'subhekabul.com', 'avapress.com', 'khaama.com', 'ariananews.co', 'ariananews.af',
    'etilaatroz.com', 'salaamtimes.com', 'bakhtarnews.com.af', 'wadsam.com',
    'afghanistantimes.af', 'afghanpaper.com', 'rawa.org', 'afghanembassy.net',
    'afghanvoice.ir', 'rahapress.af', 'ansarpress.com', 'jomhornews.com',
    'da.azadiradio.com', 'pashtoazadiradio.com', 'nunn.asia', 'kabulnow.com',
)
# Multi-service broadcasters: only the Afghan-service path counts. bbc.com/persian
# primarily serves Iran and is deliberately excluded; bbc.com/pashto does not.
AFGHAN_PATHS = (
    ('bbc.com', '/pashto'), ('bbc.co.uk', '/pashto'),
    ('dw.com', '/fa-af'), ('dw.com', '/ps-af'),
    ('rferl.org', '/afghanistan'), ('voanews.com', '/afghanistan'),
)


def host_matches(host):
    """True when host equals a known Afghan host or is a subdomain of one."""
    return any(host == name or host.endswith('.' + name) for name in AFGHAN_HOSTS)


def path_matches(host, path):
    base = host[4:] if host.startswith('www.') else host
    return any((base == h or base.endswith('.' + h)) and path.startswith(prefix)
               for h, prefix in AFGHAN_PATHS)


def classify(url):
    """Return exactly one category per document, so nothing is counted twice."""
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    if host.endswith('.af') or host == 'af':
        return 'af_tld', host
    if host_matches(host):
        return 'named_host', host
    if path_matches(host, parsed.path or ''):
        return 'named_path', host
    return 'other', host


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='fas_Arab')
    p.add_argument('--documents', type=int, default=30000)
    p.add_argument('--out', type=Path, default=ROOT / 'docs/research_af_domains.json')
    args = p.parse_args()

    stream = load_dataset('HuggingFaceFW/fineweb-2', args.config, split='train', streaming=True)
    tld = Counter()
    categories = Counter()
    hosts_by_category = {'af_tld': Counter(), 'named_host': Counter(), 'named_path': Counter()}
    af_chars = other_chars = 0
    af_samples = []
    seen = duplicates = 0
    fingerprints = set()
    started = time.time()
    for row in stream:
        url = row.get('url') or ''
        text = row.get('text') or ''
        seen += 1
        category, host = classify(url)
        tld[host.rsplit('.', 1)[-1] if '.' in host else '?'] += 1
        categories[category] += 1
        if category != 'other':
            hosts_by_category[category][host] += 1
            # Near-duplicate guard: same host and same opening 300 characters.
            fingerprint = hashlib.sha256((host + ' ' + ' '.join(text.split())[:300]).encode()).hexdigest()
            if fingerprint in fingerprints:
                duplicates += 1
                continue
            fingerprints.add(fingerprint)
            af_chars += len(text)
            if len(af_samples) < 12 and len(text) > 500:
                af_samples.append(dict(url=url[:120], category=category, chars=len(text), text=text[:300]))
        else:
            other_chars += len(text)
        if seen >= args.documents:
            break

    af_docs = categories['af_tld']
    named_docs = categories['named_host'] + categories['named_path']
    afghan_docs = af_docs + named_docs          # categories are disjoint: no double counting
    unique_afghan = afghan_docs - duplicates
    body = '\n'.join(s['text'] for s in af_samples)
    share, afghan_markers, iranian_markers, _ = dialect_score(body) if body else (0, 0, 0, [])
    report = dict(
        dataset='HuggingFaceFW/fineweb-2', config=args.config, documents_scanned=seen,
        seconds=round(time.time() - started, 1),
        method='Each document is classified into exactly one category by hostname (with label boundary) '
               'or by an Afghan-service path on a multi-service broadcaster. Near-duplicates sharing a '
               'host and opening 300 characters are counted separately and excluded from the character total.',
        af_tld_documents=af_docs, af_tld_share=round(af_docs / seen, 6),
        named_host_documents=categories['named_host'],
        named_path_documents=categories['named_path'],
        afghan_documents_total=afghan_docs, afghan_share=round(afghan_docs / seen, 6),
        near_duplicates_excluded=duplicates,
        unique_afghan_documents=unique_afghan,
        unique_afghan_share=round(unique_afghan / seen, 6),
        afghan_characters=af_chars, other_characters=other_chars,
        afghan_character_share=round(af_chars / (af_chars + other_chars), 6) if af_chars + other_chars else 0,
        top_tlds=dict(tld.most_common(12)),
        af_hosts=dict(hosts_by_category['af_tld'].most_common(20)),
        named_hosts=dict(hosts_by_category['named_host'].most_common(20)),
        named_paths=dict(hosts_by_category['named_path'].most_common(10)),
        dialect_of_afghan_sample=dict(afghan_share=round(share, 4),
                                      afghan_markers=afghan_markers, iranian_markers=iranian_markers,
                                      sample_chars=len(body)),
        projection_full_subset=dict(
            note='Linear projection of the measured unique share onto the full 58,887,122-document subset. '
                 'The stream is not shuffled, so the head may not be representative.',
            projected_afghan_documents=round(unique_afghan / seen * 58_887_122),
            projected_afghan_tokens_estimate=round(af_chars / seen * 58_887_122 / 3.3)),
        limitations='Single unshuffled pass over the head of one config. Domain is a proxy for dialect: an '
                    '.af site may publish Iranian Persian and an Afghan author may publish on a .com. The '
                    'host list is hand-collected and not exhaustive, so the true yield is likely higher. '
                    'bbc.com/persian is deliberately excluded as Iran-facing. No rights review performed.',
        samples=af_samples)
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, ensure_ascii=False, indent=2))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\nwritten:', args.out)


if __name__ == '__main__':
    main()
