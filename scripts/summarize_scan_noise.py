"""Print markdown tables of scripts/eval_scan_noise.sh results: one table per checkpoint and metric,
rows = noise condition, columns = terrain (mean over seeds; "exact" falls back to the plain terrain result).

usage: python scripts/summarize_scan_noise.py [--metric episode_length_ratio] LABEL=CKPT [LABEL=CKPT ...]
metrics: completion_rate, fall_rate, drift_rate, episode_length_ratio, root_rel_body_err, global_body_err, joint_err
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TERRAINS = ['flat', 'rough_d09', 'bumps_d10', 'tilt_d05', 'tilt_d10', 'step_d05', 'step_d10']
CONDITIONS = ['exact', 'point02', 'point05', 'bias02', 'bias05', 'drift05', 'drift10', 'drop30', 'drop60',
              'delay2', 'delay5', 'combined', 'blind_flat', 'blind_zero']
PERCENT = {'completion_rate', 'fall_rate', 'drift_rate', 'episode_length_ratio'}

parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='+')
parser.add_argument('--metric', action='append')
parser.add_argument('--dir', default='terrain_eval_v2')  # terrain_eval: older results (first episodes to finish)
args = parser.parse_args()

for run in args.runs:
    label, ckpt = run.split('=', 1) if '=' in run else (run, run)
    ckpt = Path(ckpt)
    results = [json.loads(p.read_text()) for p in (ckpt.parent / 'metrics' / args.dir / ckpt.stem).glob('*_seed*.json')]
    for metric in args.metric or ['episode_length_ratio', 'completion_rate']:
        values = defaultdict(list)
        for result in results:
            terrain, _, cond = result['name'].partition('__')
            values[terrain, cond or 'plain'].append(result[metric])
        print(f'\n**{label}: {metric}** (mean over seeds)\n')
        print('| Scan | ' + ' | '.join(TERRAINS) + ' |')
        print('|---' * (len(TERRAINS) + 1) + '|')
        for cond in CONDITIONS:
            cells = []
            for terrain in TERRAINS:
                v = values.get((terrain, cond)) or (values.get((terrain, 'plain')) if cond == 'exact' else None)
                if not v:
                    cells.append('–')
                elif metric in PERCENT:
                    cells.append(f'{np.mean(v) * 100:.1f}%')
                else:
                    cells.append(f'{np.mean(v):.3g}')
            print(f'| {cond} | ' + ' | '.join(cells) + ' |')
