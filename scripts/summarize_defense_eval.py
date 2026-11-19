"""Summarize defense_eval.py results into one table per checkpoint.

usage: python scripts/summarize_defense_eval.py PATTERN [PATTERN ...]
  PATTERN: a glob of result JSONs, e.g. 'logs/Defense/*def16*/metrics/defense_eval/*/big16_*.json'
Rows: checkpoint and eval name prefix; columns per terrain: survived / hit / block / counters landed (%), with the
episode count. The 95 % interval of a survival rate p over n episodes is about +-2 sqrt(p (1 - p) / n).
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

TERRAINS = [('flat', 0.0, 'flat'), ('step', 1.0, 'step 10 cm'), ('step', 2.0, 'step 17 cm'), ('tilt', 1.5, 'tilt 18°'),
            ('bumps', 2.0, 'bumps 8 cm')]


def main(patterns):
    rows = defaultdict(dict)
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            r = json.load(open(path))
            prefix = r['name'].rsplit('_', 2)[0]
            key = (os.path.basename(os.path.dirname(r['checkpoint'])).split('-')[1] + '/' +
                   os.path.basename(r['checkpoint'])[:-3], prefix)
            rows[key][(r['terrain'], float(r['difficulty']))] = r
    head = '| checkpoint | eval | ' + ' | '.join(t[2] for t in TERRAINS) + ' |'
    print(head)
    print('|' + '---|' * (2 + len(TERRAINS)))
    for (ckpt, prefix), res in sorted(rows.items()):
        cells = []
        for terrain, diff, _ in TERRAINS:
            r = res.get((terrain, diff))
            if r is None:
                cells.append('–')
                continue
            p, n = r['survival_rate'], r['episodes']
            ci = 200 * math.sqrt(max(p * (1 - p), 1e-4) / n)
            landed = r.get('counters', {}).get('all', {})
            landed = landed.get('landed', landed.get('land_rate')) if isinstance(landed, dict) else None
            cells.append(f"{100 * p:.1f}±{ci:.1f} / {100 * r['hit_rate']:.1f} / {100 * r['block_rate']:.0f}"
                         + (f" / {100 * landed:.1f}" if isinstance(landed, float) else ''))
        print(f'| {ckpt} | {prefix} | ' + ' | '.join(cells) + ' |')
    print('\ncells: survived % (±95 % interval) / attacks hit % / honest blocks % / counters landed %')


if __name__ == '__main__':
    main(sys.argv[1:])
