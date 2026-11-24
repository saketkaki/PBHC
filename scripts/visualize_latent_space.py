"""Visualization of the latent space for encoded observations.
This will be done to examine if there exist trends in the observations for each attack performed by the adversary.

Pull any run and visualize its encoded observations in the latent space.
The encoder for the attack observation will output a 32-dimensional vector which is compressed into a 13-dimensional vector.
The 13-dim vector consists of 8 class logits (one for each attack), 4 phase logits, and one time-to-impact logit.

Rolls out a defense checkpoint with an attack encoder (exp/defense*.yaml, AttackAwareActor), records at every step and env
the actor's latent z [32] and belief head output [13] next to the privileged labels (attack class, phase, time to impact),
and plots them. The 13 belief numbers are not treated as one 13-D vector: each slice gets its own plot.
  example_attack.png  how to read the strips: one attack zoomed in, under a labelled timeline of its phases, with
                  callouts (guessing before the onset, the moment the belief locks on, the strike remembered afterwards,
                  the time-to-impact countdown). The attack is picked automatically (misread before its onset, recognised
                  during the strike, not a calf kick, which is recognised early); --example-env picks the env to take it from
  strips.png      per example env, over time: class probabilities (heatmap, true class outlined), phase probabilities,
                  predicted vs true time to impact; dashed lines mark episode resets
  radviz.png      the 8 class probabilities as one point: the probability-weighted mean of an octagon's corners (one
                  corner per class); a confident belief sits on a corner, an unsure one near the centre. One panel per phase
  confusion.png   true vs predicted class, row-normalised, one panel per phase
  z_embedding.png z in 2-D by t-SNE, coloured by class (strike and retract steps only; guard steps in grey), by phase,
                  and by true time to impact. Only which points are near each other means anything: not the axes, and not
                  the distances between separate groups
Phases: 0 guard (before onset), 1 strike (onset to impact), 2 retract, 3 guard (after). The class label of a guard step
is the attack about to come (phase 0) or just thrown (phase 3); before the onset it cannot be seen in the keypoints.

usage (from the repo root, env `pbhc` active):
  python scripts/visualize_latent_space.py CKPT [--steps 1500] [--num-envs 64] [--seed 0] [--out DIR]
         [--terrain {flat,step,tilt,bumps} --difficulty D] [key=value ...]
  python scripts/visualize_latent_space.py --from-npz DIR/latents.npz [--out DIR]   (re-plot, no simulator)
Results go to CKPT_DIR/metrics/latent_space/<ckpt name>/seed<seed>/ (latents.npz and the PNGs) unless --out is given.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('ckpt', nargs='?')
parser.add_argument('--from-npz', default=None, help='plot a saved latents.npz instead of rolling out')
parser.add_argument('--out', default=None)
parser.add_argument('--steps', type=int, default=1500, help='control steps per env (50 Hz: 1500 = 30 s)')
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--terrain', default=None)  # flat, step, tilt or bumps: one terrain type for all envs (terrain runs)
parser.add_argument('--difficulty', type=float, default=1.0)
parser.add_argument('--strip-envs', type=int, default=3, help='example envs in strips.png')
parser.add_argument('--strip-s', type=float, default=15.0, help='seconds shown per example in strips.png')
parser.add_argument('--max-points', type=int, default=6000, help='points per panel in radviz.png and for t-SNE')
parser.add_argument('--example-env', type=int, default=None, help='env to take the attack in example_attack.png from')
args, overrides = parser.parse_known_args()
if (args.ckpt is None) == (args.from_npz is None):
    parser.error('give either CKPT or --from-npz')

import numpy as np  # noqa: E402

PHASES = ['guard (before)', 'strike', 'retract', 'guard (after)']
# reference categorical palette (dataviz skill, light mode), fixed order: one hue per attack class
CLASS_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
PHASE_COLORS = ['#d6d5cf', '#0d366b', '#6da7ec', '#52514e']  # guards in grey (light before, dark after), strike dark blue, retract light blue: apart from the class hues
GREY, INK, INK2, SURFACE = '#c3c2b7', '#0b0b0b', '#52514e', '#fcfcfb'
BLUE_RAMP = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def collect():
    """ Roll out CKPT and return the recorded arrays, each [T, N, ...]. """
    import random
    from omegaconf import OmegaConf
    import isaacgym  # noqa: F401  (must be imported before torch)
    import torch
    from hydra.utils import instantiate
    import humanoidverse.utils.config_utils  # noqa: F401  (registers the `eval` resolver)
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.envs.defense.attack_bank import ATTACKS

    ckpt = Path(args.ckpt)
    config = OmegaConf.load(ckpt.parent / 'config.yaml')
    if config.get('eval_overrides') is not None:
        config = OmegaConf.merge(config, config.eval_overrides)
    # long episodes so each example strip holds several attacks; resets still happen on falls and drift
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(
        ['headless=True', f'num_envs={args.num_envs}', f'env.config.max_episode_length_s={args.steps * 0.02 + 1}',
         'env.config.defense.debug_draw=False']
        + ([] if args.terrain is None else [
            'terrain.curriculum=False', f'terrain.terrain_types=[{args.terrain}]', 'terrain.terrain_proportions=[1.0]',
            f'terrain.fixed_difficulty={args.difficulty if args.terrain != "flat" else 0}']) + overrides))
    pre_process_config(config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = instantiate(config.env, device=args.device)
    algo = instantiate(config.algo, env=env, device=args.device, log_dir=None)
    algo.setup()
    algo.load(str(ckpt))
    algo._eval_mode()
    env.set_is_evaluating()
    policy = algo._get_inference_policy()
    actor = algo.actor
    if not hasattr(actor, 'belief_loss'):
        raise SystemExit(f'{ckpt}: the actor has no attack encoder (module_dict.actor.attack_encoder); nothing to visualize')
    module = actor.actor_module

    rec = {k: [] for k in ('z', 'belief', 'label', 'phase', 'tti', 'done')}
    with torch.no_grad():
        obs = env.reset_all()
        for step in range(args.steps):
            z, belief = module.encode(obs['actor_obs'])
            critic = obs['critic_obs']
            rec['z'].append(z.cpu().numpy())
            rec['belief'].append(belief.cpu().numpy())
            rec['label'].append(critic[:, actor.label_slice].argmax(dim=-1).cpu().numpy())
            rec['phase'].append(critic[:, actor.phase_slice].argmax(dim=-1).cpu().numpy())
            rec['tti'].append(critic[:, actor.tti_index].cpu().numpy())
            obs, _, dones, _ = env.step({'actions': policy(obs['actor_obs'])})
            rec['done'].append(dones.bool().cpu().numpy())  # the step recorded above ended its episode
            if step % 250 == 0:
                print(f'step {step}/{args.steps}')
    data = {k: np.stack(v) for k, v in rec.items()}
    data.update(dt=np.float32(env.dt), classes=np.array(ATTACKS), num_classes=module.num_classes,
                num_phases=module.num_phases, checkpoint=str(ckpt))
    return data


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def split_belief(d):
    C, P = int(d['num_classes']), int(d['num_phases'])
    b = d['belief']
    return softmax(b[..., :C]), softmax(b[..., C:C + P]), b[..., C + P]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=8)


def class_legend(fig, classes, **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='', color=c, markersize=7, label=n) for c, n in zip(CLASS_COLORS, classes)]
    fig.legend(handles=handles, ncol=len(classes), frameon=False, fontsize=8, labelcolor=INK2, **kw)


def plot_strips(d, out, plt, cmap):
    classes = list(d['classes'])
    cls_p, ph_p, tti_pred = split_belief(d)
    dt, T, N = float(d['dt']), d['label'].shape[0], d['label'].shape[1]
    L = min(T, int(args.strip_s / dt))
    envs = range(min(args.strip_envs, N))
    fig, axes = plt.subplots(3 * len(envs), 1, figsize=(12, 3.6 * len(envs)), sharex=True,
                             gridspec_kw={'height_ratios': [8, 4, 3] * len(envs), 'hspace': 0.45})
    fig.patch.set_facecolor(SURFACE)
    t = np.arange(L) * dt
    extent = (0, L * dt, -0.5, None)
    for k, e in enumerate(envs):
        a_cls, a_ph, a_tti = axes[3 * k:3 * k + 3]
        resets = t[d['done'][:L, e]] + dt
        a_cls.imshow(cls_p[:L, e].T, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest',
                     extent=extent[:3] + (len(classes) - 0.5,))
        a_cls.step(t, d['label'][:L, e], where='post', color=INK, lw=1, alpha=0.7)  # the true class
        striking = np.isin(d['phase'][:L, e], (1, 2))
        a_cls.scatter(t[striking], d['label'][:L, e][striking], s=4, color=INK, lw=0)
        a_cls.set_yticks(range(len(classes)), classes)
        a_cls.set_title(f'env {e}: attack class belief (line: true class, dots: strike and retract steps)',
                        fontsize=9, color=INK, loc='left')
        a_ph.imshow(ph_p[:L, e].T, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest',
                    extent=extent[:3] + (len(PHASES) - 0.5,))
        a_ph.step(t, d['phase'][:L, e], where='post', color=INK, lw=1, alpha=0.7)
        a_ph.set_yticks(range(len(PHASES)), PHASES)
        a_ph.set_title('phase belief (line: true phase)', fontsize=9, color=INK, loc='left')
        a_tti.plot(t, d['tti'][:L, e], color=INK2, lw=2, label='true')
        a_tti.plot(t, tti_pred[:L, e], color=BLUE_RAMP[4], lw=2, label='predicted')
        a_tti.set_ylabel('time to\nimpact [s]', fontsize=8, color=INK2)
        a_tti.set_title('time to impact: true (grey) and predicted (blue), clipped to +-1 s', fontsize=9, color=INK, loc='left')
        a_tti.grid(axis='y', color=GREY, lw=0.5, alpha=0.5)
        for ax in (a_cls, a_ph, a_tti):
            style(ax)
            for r in resets:
                ax.axvline(r, color=INK2, ls='--', lw=1)
    axes[-1].set_xlabel('time [s]', color=INK2)
    fig.colorbar(axes[0].images[0], ax=axes, fraction=0.015, pad=0.01).set_label('probability', color=INK2)
    fig.savefig(out / 'strips.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


PHASE_NOTES = ['attacker idle: the next attack\nis chosen but not visible yet',
               'onset to impact:\nthe limb moves in',
               'after the impact:\nthe limb comes back',
               'back in guard: the strike is still\nin the keypoint history']


def find_example(d, cls_p):
    """ -> (env, a, o, i, r, b): the steps at which one complete attack's guard (before), strike, retract and guard (after)
    start, and the next attack's guard (before); no reset near it. A typical attack if there is one (misread before the
    onset, recognised during the strike, not a calf kick), else the first complete one; None if there is none. """
    ph, lab, done = d['phase'], d['label'], d['done']
    pred = cls_p.argmax(-1)
    classes = list(d['classes'])
    first = None
    for e in ([args.example_env] if args.example_env is not None else range(ph.shape[1])):
        x = ph[:, e]
        starts = np.flatnonzero(np.diff(x)) + 1  # steps where a new phase starts
        for k in range(len(starts) - 4):
            a, o, i, r, b = starts[k:k + 5]
            if [x[a], x[o], x[i], x[r], x[b]] != [0, 1, 2, 3, 0] or done[max(a - 12, 0):b + 18, e].any():
                continue
            first = first or (e, a, o, i, r, b)
            if ((pred[a:o, e] == lab[a:o, e]).mean() < 0.5 and (pred[o:i, e] == lab[o:i, e]).mean() > 0.8
                    and classes[lab[o, e]] != 'calf_kick'):
                return e, a, o, i, r, b
    return first


def plot_example(d, out, plt, cmap):
    """ example_attack.png: one attack zoomed in, under a labelled timeline of its phases, with callouts. """
    classes = list(d['classes'])
    cls_p, ph_p, tti_pred = split_belief(d)
    ex = find_example(d, cls_p)
    if ex is None:
        print('example_attack.png: no complete attack without a reset nearby; skipped')
        return
    e, a, o, i, r, b = ex
    dt = float(d['dt'])
    lo, hi = max(a - 12, 0), min(b + 18, d['phase'].shape[0])
    T = lambda s: (s - o) * dt  # seconds from the onset; a phase starting at step s is drawn from T(s) - dt / 2
    t = T(np.arange(lo, hi))
    c, name = int(d['label'][o, e]), classes[int(d['label'][o, e])]
    ok = cls_p[o:i, e].argmax(-1) == c
    lock = next((o + k for k in range(len(ok)) if ok[k:].all()), None)  # names the attack from here to the impact
    box = dict(boxstyle='round,pad=0.35', fc=SURFACE, ec=GREY, alpha=0.95)
    arrow = dict(arrowstyle='->', color=INK, lw=1)
    note = dict(fontsize=8, color=INK, bbox=box, arrowprops=arrow, va='center')
    up = c < len(classes) / 2  # callouts on the side of the class panel away from the true class
    y_far, y_near = (6.3, 4.4) if up else (0.7, 2.6)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={'height_ratios': [2.4, 8, 4, 3.6], 'hspace': 0.45})
    fig.patch.set_facecolor(SURFACE)
    a_band, a_cls, a_ph, a_tti = axes
    extent = (t[0] - dt / 2, t[-1] + dt / 2)

    # the timeline: one band per phase, the previous attack's end and the next attack's start faded
    for s0, s1, p, alpha in [(lo, a, 3, 0.35), (a, o, 0, 1), (o, i, 1, 1), (i, r, 2, 1), (r, b, 3, 1), (b, hi, 0, 0.35)]:
        if s1 <= s0:
            continue
        x0, x1 = T(s0) - dt / 2, T(s1) - dt / 2
        a_band.axvspan(x0, x1, color=PHASE_COLORS[p], alpha=alpha, lw=0)
        ink = SURFACE if p in (1, 3) and alpha == 1 else INK
        if alpha == 1:
            a_band.text((x0 + x1) / 2, 0.74, PHASES[p], ha='center', va='center', fontsize=9, color=ink, weight='bold')
            a_band.text((x0 + x1) / 2, 0.34, PHASE_NOTES[p], ha='center', va='center', fontsize=7.5, color=ink, linespacing=1.3)
        else:
            a_band.text((x0 + x1) / 2, 0.5, 'previous\nattack' if s0 == lo else 'next\nattack', ha='center', va='center',
                        fontsize=7.5, color=INK2)
    a_band.set_ylim(0, 1)
    a_band.set_yticks([])
    for s in a_band.spines.values():
        s.set_visible(False)
    for s, label in ((o, 'onset'), (i, 'impact')):
        a_band.text(T(s) - dt / 2, 1.04, label, ha='center', va='bottom', fontsize=8, color=INK, weight='bold')
        for ax in axes:
            ax.axvline(T(s) - dt / 2, color=INK, lw=1, ls=(0, (4, 3)))
    for s in (a, r, b):
        for ax in axes[1:]:
            ax.axvline(T(s) - dt / 2, color=INK2, lw=0.6, alpha=0.6)

    # class belief
    a_cls.imshow(cls_p[lo:hi, e].T, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest',
                 extent=extent + (-0.5, len(classes) - 0.5))
    a_cls.step(t, d['label'][lo:hi, e], where='mid', color=INK, lw=1.2)
    a_cls.set_yticks(range(len(classes)), classes)
    a_cls.set_title('Attack class belief: darker = more probability on that attack. Line: the true attack', fontsize=9,
                    color=INK, loc='left')
    pre = cls_p[a:o, e]
    a_cls.annotate(f'Guessing. The next attack ({name}) is\nalready chosen, but the attacker has\nnot moved: the belief '
                   f'is spread\n(on the true attack {(pre.argmax(-1) == c).mean():.0%} of these steps)',
                   xy=(T((a + o) // 2), pre[(o - a) // 2].argmax()), xytext=(T(a) + 0.02, y_far), ha='left', **note)
    if lock is not None:
        a_cls.annotate(f'Locks on to {name} {1e3 * (lock - o) * dt:.0f} ms after the onset,\n'
                       f'{1e3 * (i - lock) * dt:.0f} ms before the impact', xy=(T(lock), c), xytext=(T(o) + 0.05, y_near),
                       ha='left', **note)
    a_cls.annotate('Still sure once back in guard:\nthe strike is in the keypoint\nhistory (fades as it leaves it)',
                   xy=(T(r + (b - r) // 4), c), xytext=(T(r) + 0.02, y_far), ha='left', **note)
    nxt = int(d['label'][b, e])
    a_cls.annotate(f"Next attack chosen ({classes[nxt]}{' again' if nxt == c else ''}):\nback to guessing",
                   xy=(T(b) - dt / 2, nxt), xytext=(t[-1] - 0.02, y_near), ha='right', **note)

    # phase belief
    a_ph.imshow(ph_p[lo:hi, e].T, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest',
                extent=extent + (-0.5, len(PHASES) - 0.5))
    a_ph.step(t, d['phase'][lo:hi, e], where='mid', color=INK, lw=1.2)
    a_ph.set_yticks(range(len(PHASES)), PHASES)
    a_ph.set_title('Phase belief (shading) and the true phase (line): a good belief follows the staircase', fontsize=9,
                   color=INK, loc='left')

    # time to impact
    a_tti.axhline(0, color=GREY, lw=0.8)
    a_tti.plot(t, d['tti'][lo:hi, e], color=INK2, lw=2)
    a_tti.plot(t, tti_pred[lo:hi, e], color=BLUE_RAMP[4], lw=2)
    a_tti.set_ylim(-1.35, 1.45)
    a_tti.set_ylabel('[s]', color=INK2, fontsize=8)
    a_tti.set_title('Time to impact: true (grey) counts down to 0 at the impact, then below; predicted (blue)',
                    fontsize=9, color=INK, loc='left')
    a_tti.annotate(f'flat at {tti_pred[a:o, e].mean():.1f} s: it cannot\ntell when the attack comes',
                   xy=(T((a + o) // 2), tti_pred[(a + o) // 2, e]), xytext=(T(a) + 0.02, -0.8), ha='left', **note)
    a_tti.annotate('impact: 0', xy=(T(i) - dt / 2, 0), xytext=(T(i) + 0.12, 0.95), ha='left', **note)
    a_tti.set_xlabel("time from the attack's onset [s]", color=INK2)
    for ax in axes[1:]:
        style(ax)
    a_band.tick_params(colors=INK2, labelsize=8)
    a_band.set_xlim(*extent)
    fig.suptitle(f'How to read the belief strips: one {name} thrown at the robot (env {e}), step by step',
                 color=INK, fontsize=12, y=0.95)
    fig.savefig(out / 'example_attack.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def subsample(idx, n, rng):
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)


def plot_radviz(d, out, plt, rng):
    classes = list(d['classes'])
    C = len(classes)
    cls_p, _, _ = split_belief(d)
    probs, label, phase = cls_p.reshape(-1, C), d['label'].ravel(), d['phase'].ravel()
    ang = np.pi / 2 - 2 * np.pi * np.arange(C) / C
    corners = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    xy = probs @ corners
    fig, axes = plt.subplots(1, len(PHASES), figsize=(4.2 * len(PHASES), 4.6))
    fig.patch.set_facecolor(SURFACE)
    ring = np.vstack([corners, corners[:1]])
    for p, ax in enumerate(axes):
        ax.set_facecolor(SURFACE)
        ax.plot(ring[:, 0], ring[:, 1], color=GREY, lw=1)
        for c in range(C):
            ax.plot([0, corners[c, 0]], [0, corners[c, 1]], color=GREY, lw=0.5, alpha=0.5)
            ax.text(*(corners[c] * 1.13), classes[c], ha='center', va='center', fontsize=7, color=INK2)
        idx = subsample(np.flatnonzero(phase == p), args.max_points, rng)
        rng.shuffle(idx)  # no class drawn last on top of the others
        ax.scatter(xy[idx, 0], xy[idx, 1], s=8, c=np.array(CLASS_COLORS)[label[idx]], alpha=0.35, lw=0)
        acc = (probs[phase == p].argmax(-1) == label[phase == p]).mean() if (phase == p).any() else float('nan')
        ax.set_title(f'{PHASES[p]}\n{(phase == p).sum()} steps, class acc {acc:.0%}', fontsize=9, color=INK)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect('equal')
        ax.axis('off')
    class_legend(fig, classes, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Attack class belief (colour: true class; corner: confident, centre: unsure)', color=INK, fontsize=11)
    fig.savefig(out / 'radviz.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(d, out, plt, cmap):
    classes = list(d['classes'])
    C = len(classes)
    cls_p, _, _ = split_belief(d)
    pred, label, phase = cls_p.argmax(-1).ravel(), d['label'].ravel(), d['phase'].ravel()
    fig, axes = plt.subplots(1, len(PHASES), figsize=(4.4 * len(PHASES), 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for p, ax in enumerate(axes):
        m = np.zeros((C, C))
        sel = phase == p
        np.add.at(m, (label[sel], pred[sel]), 1)
        m /= np.maximum(m.sum(1, keepdims=True), 1)
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        for i in range(C):
            for j in range(C):
                if m[i, j] >= 0.1:
                    ax.text(j, i, f'{m[i, j]:.0%}'.replace('%', ''), ha='center', va='center', fontsize=6,
                            color=SURFACE if m[i, j] > 0.55 else INK)
        ax.set_xticks(range(C), classes, rotation=60, ha='right', fontsize=7)
        ax.set_yticks(range(C), classes, fontsize=7)
        ax.set_xlabel('predicted', color=INK2, fontsize=8)
        ax.set_title(f'{PHASES[p]} ({sel.sum()} steps)', fontsize=9, color=INK)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel('true', color=INK2, fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01).set_label('share of the true class (%, shown from 10)', color=INK2)
    fig.savefig(out / 'confusion.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def plot_z(d, out, plt, cmap, rng):
    from sklearn.manifold import TSNE
    from matplotlib.lines import Line2D
    classes = list(d['classes'])
    z = d['z'].reshape(-1, d['z'].shape[-1])
    label, phase, tti = d['label'].ravel(), d['phase'].ravel(), d['tti'].ravel()
    # balanced sample: guard steps outnumber strike and retract steps several times over
    idx = np.concatenate([subsample(np.flatnonzero(phase == p), args.max_points // len(PHASES), rng) for p in range(len(PHASES))])
    rng.shuffle(idx)
    zs = (z[idx] - z.mean(0)) / (z.std(0) + 1e-6)
    print(f't-SNE on {len(idx)} points ...')
    xy = TSNE(2, perplexity=40, init='pca', random_state=args.seed).fit_transform(zs)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    fig.patch.set_facecolor(SURFACE)
    lab, ph, tt = label[idx], phase[idx], tti[idx]
    attack = np.isin(ph, (1, 2))
    a_cls, a_ph, a_tti = axes
    a_cls.scatter(*xy[~attack].T, s=5, color=GREY, alpha=0.3, lw=0)
    a_cls.scatter(*xy[attack].T, s=7, c=np.array(CLASS_COLORS)[lab[attack]], alpha=0.6, lw=0)
    a_cls.set_title('by attack (strike and retract steps; guard steps in grey)', fontsize=9, color=INK, loc='left')
    a_ph.scatter(*xy.T, s=6, c=np.array(PHASE_COLORS)[ph], alpha=0.5, lw=0)
    a_ph.set_title('by phase', fontsize=9, color=INK, loc='left')
    a_ph.legend(handles=[Line2D([], [], marker='o', ls='', color=c, markersize=7, label=n) for c, n in zip(PHASE_COLORS, PHASES)],
                frameon=False, fontsize=8, labelcolor=INK2, loc='best')
    sc = a_tti.scatter(*xy.T, s=6, c=tt, cmap=cmap.reversed(), vmin=-1, vmax=1, alpha=0.6, lw=0)
    a_tti.set_title('by true time to impact (clipped to +-1 s; light: impact ahead, dark: impact past)', fontsize=9, color=INK, loc='left')
    fig.colorbar(sc, ax=a_tti, fraction=0.04).set_label('time to impact [s]', color=INK2)
    for ax in axes:
        style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    class_legend(fig, classes, loc='lower center', bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'Attack encoder latent z ({z.shape[-1]}-D) by t-SNE: one dot per step ({len(idx)} steps, balanced over the '
                 f'phases); nearby dots had similar z.\nAxes and the distances between separate groups carry no meaning.',
                 color=INK, fontsize=11)
    fig.savefig(out / 'z_embedding.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main():
    if args.from_npz is not None:
        npz = Path(args.from_npz)
        d = dict(np.load(npz, allow_pickle=False))
        out = Path(args.out) if args.out else npz.parent
    else:
        d = collect()
        ckpt = Path(args.ckpt)
        out = Path(args.out) if args.out else ckpt.parent / 'metrics' / 'latent_space' / ckpt.stem / f'seed{args.seed}'
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'latents.npz', **d)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blue_ramp', BLUE_RAMP)
    point_cmap = LinearSegmentedColormap.from_list('blue_ramp_points', BLUE_RAMP[2:])  # no near-white points
    rng = np.random.default_rng(args.seed)

    cls_p, ph_p, tti_pred = split_belief(d)
    striking = d['phase'] == 1
    print(f"{d['label'].size} steps ({d['label'].shape[1]} envs x {d['label'].shape[0]}) | class acc "
          f"{(cls_p.argmax(-1) == d['label']).mean():.1%} (strike {(cls_p.argmax(-1) == d['label'])[striking].mean():.1%}) | "
          f"phase acc {(ph_p.argmax(-1) == d['phase']).mean():.1%} | "
          f"tti MAE during strike {1e3 * np.abs(tti_pred - d['tti'])[striking].mean():.0f} ms")
    plot_example(d, out, plt, cmap)
    plot_strips(d, out, plt, cmap)
    plot_radviz(d, out, plt, rng)
    plot_confusion(d, out, plt, cmap)
    plot_z(d, out, plt, point_cmap, rng)
    print(f'wrote {out}/{{latents.npz, example_attack.png, strips.png, radviz.png, confusion.png, z_embedding.png}}')


if __name__ == '__main__':
    main()
