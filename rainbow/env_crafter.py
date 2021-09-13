# -*- coding: utf-8 -*-
import pathlib
from collections import deque

import crafter
import numpy as np
import torch

class Env():

  def __init__(self, args):
    self.device = args.device
    env = crafter.Env()
    env = crafter.Recorder(
        env, pathlib.Path(args.logdir) / 'crafter-episodes',
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    env = ResizeImage(env)
    env = GrayScale(env)
    self.env = env
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)

  def reset(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))
    obs = self.env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
    self.state_buffer.append(obs)
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    obs, reward, done, _ = self.env.step(action)
    obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
    self.state_buffer.append(obs)
    return torch.stack(list(self.state_buffer), 0), reward, done

  def action_space(self):
    return self.env.action_space.n

  def train(self):
    pass

  def test(self):
    pass


class GrayScale:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = obs.mean(-1)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = obs.mean(-1)
    return obs


class ResizeImage:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = self._resize(obs)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = self._resize(obs)
    return obs

  def _resize(self, image):
    from PIL import Image
    image = Image.fromarray(image)
    image = image.resize((84, 84), Image.NEAREST)
    image = np.array(image)
    return image
