"""Print a markdown table of scripts/terrain_eval.py results (mean ± std over seeds) for several checkpoints.

usage: python scripts/summarize_terrain_eval.py [--metric completion_rate] LABEL=CKPT [LABEL=CKPT ...]
metrics: completion_rate, fall_rate, drift_rate, episode_length_ratio, root_rel_body_err, global_body_err, joint_err
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TERRAINS = ['flat', 'rough_d09', 'tilt_d05', 'tilt_d10', 'step_d05', 'step_d10']
PERCENT = {'completion_rate', 'fall_rate', 'drift_rate', 'episode_length_ratio'}

parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='+')
parser.add_argument('--metric', action='append')
args = parser.parse_args()

for metric in args.metric or ['completion_rate', 'episode_length_ratio', 'root_rel_body_err']:
    print(f'\n**{metric}** (mean ± std over seeds)\n')
    print('| Model | ' + ' | '.join(TERRAINS) + ' |')
    print('|---' * (len(TERRAINS) + 1) + '|')
    for run in args.runs:
        label, ckpt = run.split('=', 1) if '=' in run else (run, run)
        ckpt = Path(ckpt)
        values = defaultdict(list)
        for path in (ckpt.parent / 'metrics' / 'terrain_eval' / ckpt.stem).glob('*_seed*.json'):
            result = json.loads(path.read_text())
            values[result['name']].append(result[metric])
        cells = []
        for terrain in TERRAINS:
            v = np.array(values.get(terrain, []))
            if len(v) == 0:
                cells.append('–')
            elif metric in PERCENT:
                cells.append(f'{v.mean() * 100:.1f} ± {v.std() * 100:.1f}%')
            else:
                cells.append(f'{v.mean():.3g} ± {v.std():.2g}')
        print(f'| {label} | ' + ' | '.join(cells) + ' |')
