"""Print markdown tables of scripts/eval_difficulty_sweep.sh results: per terrain type, rows = checkpoint,
columns = difficulty (mean ± std over seeds). With --step-edges, also the step fall rate by how close the reference
feet come to the step edge, and by whether the edge passes between the feet; and (results of terrain_eval.py with the
edge-direction diagnostic) by where the raised side is relative to the start heading, by how many reference feet
start on it, and how often a shin touches something in falls and in other episodes. With --drift (results of
terrain_eval.py with the drift breakdown), how drift episodes end: which body is furthest off, whether horizontally or
vertically, the heading error, the pelvis tilt, how fast the error grew, and on steps whether the robot's feet were on another level
than the reference's.

usage: python scripts/summarize_difficulty_sweep.py [--metric fall_rate] [--step-edges] [--drift] LABEL=CKPT [LABEL=CKPT ...]
metrics: completion_rate, fall_rate, drift_rate, episode_length_ratio, root_rel_body_err, global_body_err, joint_err
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PERCENT = {'completion_rate', 'fall_rate', 'drift_rate', 'episode_length_ratio'}
LABELS = {'step': lambda d: f'{100 * (0.03 + 0.07 * d):.1f} cm', 'tilt': lambda d: f'{12 * d:.0f}°',
          'bumps': lambda d: f'{4 * d:.0f} cm', 'rough': lambda d: f'd {d:.1f}', 'flat': lambda d: ''}
EDGE_BINS = [0, 0.05, 0.1, 0.2, np.inf]  # [m] closest approach of a reference foot (ankle) to the step edge

parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='+')
parser.add_argument('--metric', action='append')
parser.add_argument('--step-edges', action='store_true')
parser.add_argument('--drift', action='store_true')
parser.add_argument('--dir', default='terrain_eval_v2')
args = parser.parse_args()


def load(ckpt):
    """{(type, difficulty): [result, ...]}"""
    results = defaultdict(list)
    for path in (ckpt.parent / 'metrics' / args.dir / ckpt.stem).glob('*_seed*.json'):
        result = json.loads(path.read_text())
        match = re.fullmatch(r'([a-z]+)(?:_d(\d+))?', result['name'])
        if match:
            results[match[1], int(match[2]) / 100 if match[2] else 0.].append(result)
    return results


runs = []
for run in args.runs:
    label, ckpt = run.split('=', 1) if '=' in run else (run, run)
    runs.append((label, load(Path(ckpt))))
keys = sorted({key for _, results in runs for key in results})

for metric in args.metric or ['fall_rate', 'episode_length_ratio']:
    for type_ in dict.fromkeys(t for t, _ in keys):
        diffs = [d for t, d in keys if t == type_]
        print(f'\n**{type_}: {metric}** (mean ± std over seeds)\n')
        print('| Model | ' + ' | '.join(f'd {d:g} {LABELS[type_](d)}'.strip() for d in diffs) + ' |')
        print('|---' * (len(diffs) + 1) + '|')
        for label, results in runs:
            cells = []
            for d in diffs:
                v = np.array([r[metric] for r in results.get((type_, d), [])])
                if len(v) == 0:
                    cells.append('–')
                elif metric in PERCENT:
                    cells.append(f'{v.mean() * 100:.1f} ± {v.std() * 100:.1f}%')
                else:
                    cells.append(f'{v.mean():.3g} ± {v.std():.2g}')
            print(f'| {label} | ' + ' | '.join(cells) + ' |')

if args.step_edges:
    for label, results in runs:
        print(f'\n**{label}: step fall rate by closest foot-edge distance** (all seeds; episodes in brackets)\n')
        bins = [f'{lo * 100:g}–{hi * 100:g} cm' if hi < np.inf else f'> {lo * 100:g} cm'
                for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:])]
        print('| Difficulty | ' + ' | '.join(bins) + ' | edge between feet | edge not between feet |')
        print('|---' * (len(bins) + 3) + '|')
        directions = {}
        for (type_, d), rs in sorted(results.items()):
            episodes = [e for r in rs for e in r.get('step_episodes', [])]
            if type_ != 'step' or not episodes:
                continue
            fall = np.array([e['outcome'] == 'fall' for e in episodes])
            dist = np.array([min(e['min_foot_edge_dist']) for e in episodes])
            straddle = np.array([e['straddle_frac'] > 0.5 for e in episodes])

            def cell(mask):
                return f'{fall[mask].mean() * 100:.0f}% ({mask.sum()})' if mask.any() else '–'
            cells = [cell((dist >= lo) & (dist < hi)) for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:])]
            print(f'| {d:g} ({LABELS["step"](d)}) | ' + ' | '.join(cells + [cell(straddle), cell(~straddle)]) + ' |')

            if 'raised_bearing_deg' not in episodes[0]:
                continue
            bearing = np.array([e['raised_bearing_deg'] for e in episodes])
            high = np.array([e['start_feet_high'] for e in episodes])
            shin = np.array([e['shin_contact_frac'] for e in episodes])
            ended = np.array([e['outcome'] != 'completed' for e in episodes])

            def cell2(mask):
                return f'{fall[mask].mean() * 100:.0f} / {ended[mask].mean() * 100:.0f}% ({mask.sum()})' if mask.any() else '–'
            directions[d] = [cell2(np.abs(bearing) < 45), cell2((bearing >= 45) & (bearing < 135)), cell2(np.abs(bearing) >= 135),
                             cell2((bearing > -135) & (bearing <= -45))] + [cell2(high == k) for k in (0, 1, 2)] + \
                [f'{shin[fall].mean() * 100:.0f}% / {shin[~fall].mean() * 100:.0f}%' if fall.any() and (~fall).any() else '–']
        if directions:
            print(f'\n**{label}: step fall / not-completed rate by raised side (start heading frame) and reference feet '
                  f'starting on it** (episodes in brackets)\n')
            print('| Difficulty | ahead | left | behind | right | 0 feet up | 1 foot up | 2 feet up | shin contact, falls / others |')
            print('|---' * 9 + '|')
            for d, cells in directions.items():
                print(f'| {d:g} ({LABELS["step"](d)}) | ' + ' | '.join(cells) + ' |')


if args.drift:
    for type_, d in keys:
        if not any(r.get('drift_episodes') for _, results in runs for r in results.get((type_, d), [])):
            continue
        print(f'\n**{type_} {LABELS[type_](d) or ""}: how drift episodes end** (all seeds; medians; error one step before '
              f'the end)\n')
        step = type_ == 'step'
        print('| Model | drift episodes | furthest body | body error xy / z [m] | mostly vertical | root error xy / z [m] | '
              'heading error | pelvis tilt | half to full threshold | sudden (< 0.5 s)' + (' | feet on another level than the reference'
                                                                            if step else '') + ' |')
        print('|---' * (11 if step else 10) + '|')
        for label, results in runs:
            episodes = [e for r in results.get((type_, d), []) for e in r.get('drift_episodes', [])]
            if not episodes:
                print(f'| {label} | 0 |' + ' – |' * (9 if step else 8))
                continue
            med = lambda k: float(np.median([e[k] for e in episodes]))
            bodies = defaultdict(int)
            for e in episodes:
                bodies[e['body'].replace('_link', '')] += 1
            top = ', '.join(f'{b} {100 * n / len(episodes):.0f}%' for b, n in sorted(bodies.items(), key=lambda x: -x[1])[:2])
            vertical = np.mean([e['z'] > e['xy'] for e in episodes])
            sudden = np.mean([e['grow_s'] < 0.5 for e in episodes])
            cells = [str(len(episodes)), top, f'{med("xy"):.2f} / {med("z"):.2f}', f'{vertical * 100:.0f}%',
                     f'{med("root_xy"):.2f} / {med("root_z"):.2f}', f'{med("yaw_deg"):.0f}°',
                     f'{med("tilt_deg"):.0f}°' if 'tilt_deg' in episodes[0] else '–', f'{med("grow_s"):.1f} s',
                     f'{sudden * 100:.0f}%']
            if step:
                cells.append(f'{np.mean([e["robot_feet_high"] != e["ref_feet_high"] for e in episodes]) * 100:.0f}%')
            print(f'| {label} | ' + ' | '.join(cells) + ' |')
