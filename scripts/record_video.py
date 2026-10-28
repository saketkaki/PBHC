"""Record a fixed-length video of a checkpoint in the Isaac Gym viewer, optionally upload it to W&B.

Unlike the viewer's own recorder (key R), every control step is rendered exactly once, so the video
plays at real time (50 fps) and clips of different policies line up frame by frame. The camera is
fixed on the robot's start position.

usage (from the repo root, env `pbhc` active, a display available):
  python scripts/record_video.py record CKPT --out clip.mp4 [--seconds 10] [--label TEXT]
         [--wandb-run ENTITY/PROJECT/RUN_ID] [--wandb-key videos/step_d10] [--merge cfg.yaml ...] [key=value ...]
  python scripts/record_video.py stack OUT.mp4 IN1.mp4 IN2.mp4 ...   # side by side

  e.g. step terrain at difficulty 1.0 (the merge/overrides are only needed for checkpoints not trained on it):
  python scripts/record_video.py record logs/MotionTracking/<run>/model_3000.pt --out step_d10.mp4 --label "C, step 10 cm" \
      --merge humanoidverse/config/terrain/terrain_kungfu.yaml "terrain.terrain_types=[step]" \
      "terrain.terrain_proportions=[1.0]" terrain.fixed_difficulty=1.0
"""
import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def encode(frames_dir, out, fps=50):
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', str(fps), '-i', str(frames_dir / '%05d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True)


def stack(out, inputs, height=540):
    filters = ''.join(f'[{i}:v]scale=-2:{height}[v{i}];' for i in range(len(inputs)))
    filters += ''.join(f'[v{i}]' for i in range(len(inputs))) + f'hstack=inputs={len(inputs)}'
    cmd = ['ffmpeg', '-loglevel', 'error', '-y']
    for path in inputs:
        cmd += ['-i', str(path)]
    subprocess.run(cmd + ['-filter_complex', filters, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True)
    print(f'wrote {out}')


def record(args, overrides):
    from omegaconf import OmegaConf
    import isaacgym  # noqa: F401  (must be imported before torch)
    from isaacgym import gymapi
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from PIL import Image, ImageDraw, ImageFont
    import humanoidverse.utils.config_utils  # noqa: F401  (registers the `eval` resolver)
    from humanoidverse.utils.helpers import pre_process_config

    ckpt = Path(args.ckpt)
    config = OmegaConf.load(ckpt.parent / 'config.yaml')
    if config.get('eval_overrides') is not None:
        config = OmegaConf.merge(config, config.eval_overrides)
    for path in args.merge:
        cfg = OmegaConf.load(path)
        cfg.pop('defaults', None)
        for key in cfg:  # replace whole groups (e.g. `terrain`), like a hydra group override
            config[key] = cfg[key]
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(['headless=False', 'num_envs=1'] + overrides))
    pre_process_config(config)
    frames_dir = Path(tempfile.mkdtemp(prefix='pbhc_frames_'))
    config.env.config.save_rendering_dir = str(frames_dir)  # required by the viewer setup; its own recorder is not used

    # terrain generation uses numpy, domain randomisation and tile assignment torch: seed both so that
    # every checkpoint is filmed on the same terrain tile with the same step edge
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

    sim = env.simulator
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 28)
    state = {'frame': 0, 'target': None}

    def render(sync_frame_time=True, _realtime=True):  # replaces the real-time viewer loop: one frame per control step
        if sim.gym.query_viewer_has_closed(sim.viewer):
            sys.exit()
        sim.gym.fetch_results(sim.sim, True)
        if state['target'] is None:  # fix the camera on the start position
            root = sim.robot_root_states[0, :3].tolist()
            state['target'] = gymapi.Vec3(root[0], root[1], root[2] - 0.2)
            state['cam'] = gymapi.Vec3(root[0] + args.cam_offset[0], root[1] + args.cam_offset[1], root[2] + args.cam_offset[2])
        sim.gym.viewer_camera_look_at(sim.viewer, None, state['cam'], state['target'])
        sim.gym.step_graphics(sim.sim)
        sim.gym.draw_viewer(sim.viewer, sim.sim, True)
        sim.gym.poll_viewer_events(sim.viewer)
        sim.gym.write_viewer_image_to_file(sim.viewer, str(frames_dir / f'{state["frame"]:05d}.png'))
        state['frame'] += 1
    sim.render = render

    num_steps = int(round(args.seconds / env.dt))
    falls, episodes, ends = 0, 0, []
    with torch.no_grad():
        obs = env.reset_all()
        for step in range(num_steps):
            obs, _, dones, extras = env.step({'actions': policy(obs['actor_obs'])})
            if dones[0]:
                episodes += 1
                reasons = [k for k, v in env.reset_buf_terminate_by.items() if bool(v[0])]
                ends.append(f'{(step + 1) * env.dt:.2f}s:{"+".join(reasons) or "?"}')
                if 'motion_end' not in reasons:
                    falls += 1

    # label every frame, then encode
    caption = f'{args.label}   falls: {falls} in {args.seconds:g} s' if args.label else f'falls: {falls} in {args.seconds:g} s'
    for png in sorted(frames_dir.glob('*.png')):
        img = Image.open(png).convert('RGB')
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
        draw.text((12, 6), caption, fill=(255, 255, 255), font=font)
        img.save(png)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    encode(frames_dir, out)
    shutil.rmtree(frames_dir)
    print(f'wrote {out}: {state["frame"]} frames, {episodes} episode ends, {falls} falls (episode ends before the motion end)')
    print('episode ends (time: reason):', ' '.join(ends))

    if args.wandb_run:
        import wandb
        entity, project, run_id = args.wandb_run.split('/')
        run = wandb.init(entity=entity, project=project, id=run_id, resume='allow')
        run.log({args.wandb_key: wandb.Video(str(out), fps=50, format='mp4', caption=caption)})
        run.finish()
        print(f'uploaded to {args.wandb_run} as {args.wandb_key}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    rec = sub.add_parser('record')
    rec.add_argument('ckpt')
    rec.add_argument('--out', required=True)
    rec.add_argument('--seconds', type=float, default=10.0)
    rec.add_argument('--label', default='')
    rec.add_argument('--seed', type=int, default=0)
    rec.add_argument('--device', default='cuda:0')
    rec.add_argument('--cam-offset', type=float, nargs=3, default=[1.8, -1.8, 0.6], help='camera position relative to the start root [m]')
    rec.add_argument('--merge', action='append', default=[])
    rec.add_argument('--wandb-run', default=None, help='ENTITY/PROJECT/RUN_ID to attach the video to')
    rec.add_argument('--wandb-key', default='videos/eval')
    st = sub.add_parser('stack')
    st.add_argument('out')
    st.add_argument('inputs', nargs='+')
    args, overrides = parser.parse_known_args()
    if args.mode == 'stack':
        stack(args.out, args.inputs)
    else:
        record(args, overrides)
