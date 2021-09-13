import pathlib
import random
import argparse

import crafter
import tqdm

parser = argparse.ArgumentParser()
boolean = lambda x: bool(['false', 'true'].index(x.lower()))
parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()

env = crafter.Env()
env = crafter.Recorder(
    env, pathlib.Path(args.logdir) / 'crafter-episodes',
    save_stats=True,
    save_video=False,
    save_episode=False,
)

num_actions = env.action_space.n

done = True
for _ in tqdm.trange(int(args.steps), smoothing=0):
  if done:
    env.reset()
  action = random.randint(0, num_actions - 1)
  _, _, done, _ = env.step(action)
