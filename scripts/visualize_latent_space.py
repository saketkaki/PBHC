"""Visualization of the latent space for encoded observations.
This will be done to examine if there exist trends in the observations for each attack performed by the adversary.

Pull any run and visualize its encoded observations in the latent space.
The encoder for the attack observation will output a 32-dimensional vector which is compressed into a 13-dimensional vector.
The 13-dim vector consists of 8 class logits (one for each attack), 4 phase logits, and one time-to-impact logit.

Rolls out a defense checkpoint with an attack encoder (exp/defense*.yaml, AttackAwareActor), records at every step and env
the actor's latent z [32] and belief head output [13] next to the privileged labels (attack class, phase, time to impact),
and plots them. The 13 belief numbers are not treated as one 13-D vector: each slice gets its own plot.
  strips.png      per example env, over time: class probabilities (heatmap, true class outlined), phase probabilities,
                  predicted vs true time to impact; dashed lines mark episode resets
  radviz.png      the 8 class probabilities as one point: the probability-weighted mean of an octagon's corners (one
                  corner per class); a confident belief sits on a corner, an unsure one near the centre. One panel per phase
  confusion.png   true vs predicted class, row-normalised, one panel per phase
  z_embedding.png z in 2-D by PCA and t-SNE, coloured by class (strike and retract steps only; guard steps in grey), by
                  phase, and by true time to impact
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
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from matplotlib.lines import Line2D
    classes = list(d['classes'])
    z = d['z'].reshape(-1, d['z'].shape[-1])
    label, phase, tti = d['label'].ravel(), d['phase'].ravel(), d['tti'].ravel()
    # balanced sample: guard steps outnumber strike and retract steps several times over
    idx = np.concatenate([subsample(np.flatnonzero(phase == p), args.max_points // len(PHASES), rng) for p in range(len(PHASES))])
    rng.shuffle(idx)
    zs = (z[idx] - z.mean(0)) / (z.std(0) + 1e-6)
    pca = PCA(2).fit(zs)
    emb = {f'PCA (explains {pca.explained_variance_ratio_.sum():.0%} of the variance)': pca.transform(zs)}
    print(f't-SNE on {len(idx)} points ...')
    emb['t-SNE'] = TSNE(2, perplexity=40, init='pca', random_state=args.seed).fit_transform(zs)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    fig.patch.set_facecolor(SURFACE)
    lab, ph, tt = label[idx], phase[idx], tti[idx]
    attack = np.isin(ph, (1, 2))
    for r, (name, xy) in enumerate(emb.items()):
        a_cls, a_ph, a_tti = axes[r]
        a_cls.scatter(*xy[~attack].T, s=5, color=GREY, alpha=0.3, lw=0)
        a_cls.scatter(*xy[attack].T, s=7, c=np.array(CLASS_COLORS)[lab[attack]], alpha=0.6, lw=0)
        a_cls.set_title(f'{name}\nby class (strike and retract; guard in grey)', fontsize=9, color=INK, loc='left')
        a_ph.scatter(*xy.T, s=6, c=np.array(PHASE_COLORS)[ph], alpha=0.5, lw=0)
        a_ph.set_title('by phase', fontsize=9, color=INK, loc='left')
        a_ph.legend(handles=[Line2D([], [], marker='o', ls='', color=c, markersize=7, label=n) for c, n in zip(PHASE_COLORS, PHASES)],
                    frameon=False, fontsize=8, labelcolor=INK2, loc='best')
        sc = a_tti.scatter(*xy.T, s=6, c=tt, cmap=cmap.reversed(), vmin=-1, vmax=1, alpha=0.6, lw=0)
        a_tti.set_title('by true time to impact (clipped to +-1 s; light: impact ahead, dark: impact past)', fontsize=9, color=INK, loc='left')
        fig.colorbar(sc, ax=a_tti, fraction=0.04).set_label('time to impact [s]', color=INK2)
        for ax in axes[r]:
            style(ax)
            ax.set_xticks([])
            ax.set_yticks([])
    class_legend(fig, classes, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'Attack encoder latent z ({z.shape[-1]}-D), {len(idx)} steps balanced over the phases', color=INK, fontsize=12)
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
    plot_strips(d, out, plt, cmap)
    plot_radviz(d, out, plt, rng)
    plot_confusion(d, out, plt, cmap)
    plot_z(d, out, plt, point_cmap, rng)
    print(f'wrote {out}/{{latents.npz, strips.png, radviz.png, confusion.png, z_embedding.png}}')


if __name__ == '__main__':
    main()
