"""Compare checkpoint scorecards, flag regressions, render an HTML dashboard."""
import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / 'outputs/scorecards'

# metric path -> (label, better direction, regression threshold as a fraction)
METRICS = [
    ('tier2_language.dari_heldout.perplexity', 'Dari perplexity', 'lower', 0.02),
    ('tier2_language.science_heldout.perplexity', 'Science perplexity', 'lower', 0.02),
    ('tier2_language.pashto_heldout.perplexity', 'Pashto perplexity', 'lower', 0.02),
    ('tier2_language.generation.real_word_rate', 'Real-word rate', 'higher', 0.02),
    ('tier2_language.generation.dialect_afghan_share', 'Afghan dialect share', 'higher', 0.05),
    ('tier2_language.generation.distinct_4gram', 'Distinct 4-grams', 'higher', 0.05),
    ('tier2_language.generation.longest_repeat', 'Longest repeat', 'lower', 0.25),
    ('tier3_instruction.accuracy', 'Maths accuracy', 'higher', 0.02),
]


def dig(card, path):
    node = card
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, (int, float)) else None


def verdict(previous, current, direction, threshold):
    if previous is None or current is None:
        return 'n/a', ''
    if previous == 0:
        return ('same', '') if current == 0 else ('changed', '')
    change = (current - previous) / abs(previous)
    improved = change < 0 if direction == 'lower' else change > 0
    if abs(change) < threshold:
        return 'flat', f'{change:+.1%}'
    return ('better' if improved else 'REGRESSION'), f'{change:+.1%}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cards-dir', type=Path, default=CARDS)
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/scorecards/dashboard.html')
    args = p.parse_args()

    files = sorted(f for f in args.cards_dir.glob('*.json'))
    cards = []
    for f in files:
        try:
            card = json.loads(f.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue
        if 'tier2_language' in card:
            cards.append((f.stem, card))
    if not cards:
        raise SystemExit(f'No scorecards found in {args.cards_dir}')
    cards.sort(key=lambda c: c[1]['tier1_pipeline'].get('step') or 0)

    rows, regressions = [], []
    for label, path, direction, threshold in [(m[1], m[0], m[2], m[3]) for m in METRICS]:
        values = [dig(card, path) for _, card in cards]
        cells = []
        for i, value in enumerate(values):
            if i == 0 or value is None or values[i - 1] is None:
                cells.append((value, '', ''))
                continue
            state, change = verdict(values[i - 1], value, direction, threshold)
            if state == 'REGRESSION':
                regressions.append(f'{label}: {values[i-1]} -> {value} ({change}) at {cards[i][0]}')
            cells.append((value, state, change))
        rows.append((label, direction, cells))

    def fmt(value):
        return '—' if value is None else (f'{value:.4g}' if isinstance(value, float) else str(value))

    head = ''.join(f'<th>{html.escape(name)}<br><small>step '
                   f'{card["tier1_pipeline"].get("step") or "?"}</small></th>' for name, card in cards)
    body = ''
    for label, direction, cells in rows:
        tds = ''.join(
            f'<td class="{state.lower().replace(" ", "")}">{fmt(value)}'
            f'{f"<br><small>{change}</small>" if change else ""}</td>'
            for value, state, change in cells)
        body += f'<tr><th>{html.escape(label)}<br><small>{direction} is better</small></th>{tds}</tr>'

    banner = ('<div class="ok">No regressions detected.</div>' if not regressions else
              '<div class="bad"><b>REGRESSIONS</b><ul>' +
              ''.join(f'<li>{html.escape(r)}</li>' for r in regressions) + '</ul></div>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>AMC checkpoint comparison</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#fbfbfb;color:#111}}
table{{border-collapse:collapse;margin-top:1rem}}
th,td{{border:1px solid #ddd;padding:.5rem .8rem;text-align:right}}
th{{background:#f2f2f2;text-align:left}} small{{color:#666;font-weight:normal}}
.better{{background:#e7f6e7}} .regression{{background:#fde2e1;font-weight:bold}} .flat{{background:#fff}}
.ok{{background:#e7f6e7;padding:.8rem;border-radius:6px}}
.bad{{background:#fde2e1;padding:.8rem;border-radius:6px}}
.note{{margin-top:1.5rem;padding:.8rem;background:#fff8e1;border-radius:6px;max-width:60rem}}
</style></head><body>
<h1>AMC checkpoint comparison</h1>{banner}
<table><tr><th>Metric</th>{head}</tr>{body}</table>
<div class="note"><b>Reading this honestly.</b> These are tier 1–3 measurements only.
Lower perplexity means better next-token prediction on held-out text; it does <i>not</i> mean the
model is coherent, truthful, or useful to a student. Tier 4 (educational usefulness) is not on this
page because it cannot be automated — it requires a native-speaker rubric
(<code>docs/EVALUATION.md</code>). A green row is necessary, never sufficient.</div>
</body></html>"""
    args.out.write_text(page, encoding='utf-8')
    print(f'Compared {len(cards)} scorecard(s): ' + ', '.join(n for n, _ in cards))
    print(f'Regressions: {len(regressions)}')
    for r in regressions:
        print('  ', r)
    print('dashboard:', args.out)


if __name__ == '__main__':
    main()
