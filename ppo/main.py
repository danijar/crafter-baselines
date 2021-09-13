import argparse
import pathlib

import chunkedfile
import crafter
import stable_baselines3

chunkedfile.patch_pathlib_append(3600)

parser = argparse.ArgumentParser()
boolean = lambda x: bool(['False', 'True'].index(x))
parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--steps', type=float, default=5e6)
args = parser.parse_args()

env = crafter.Env()
env = crafter.Recorder(
    env, pathlib.Path(args.logdir) / 'crafter-episodes',
    save_stats=True,
    save_video=False,
    save_episode=False,
)

model = stable_baselines3.PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=args.steps)
