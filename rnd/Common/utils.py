import pathlib
import numpy as np
import gym
import crafter
from PIL import Image


def mean_of_list(func):
    def function_wrapper(*args, **kwargs):
        lists = func(*args, **kwargs)
        return [sum(list) / len(list) for list in lists[:-4]] + [explained_variance(lists[-4], lists[-3])] + \
               [explained_variance(lists[-2], lists[-1])]

    return function_wrapper


def preprocessing(img):
    # import cv2
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(img)
    img = img.resize((84, 84), Image.NEAREST)
    img = np.array(img)
    img = img.mean(-1)
    assert img.shape == (84, 84), img.shape
    return img


def stack_states(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=0)
    else:
        stacked_frames = stacked_frames[1:, ...]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=0)], axis=0)
    return stacked_frames


# Calculates if value function is a good predictor of the returns (ev > 1)
# or if it's just worse than predicting nothing (ev =< 0)
def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def make_crafter(logdir, reward):
    env = crafter.Env(reward=reward)
    env = crafter.Recorder(
        env, pathlib.Path(logdir) / 'crafter-episodes',
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    # env = PadImage(env)
    # env = GrayScale(env)
    # env = TransposeImage(env)
    env._max_episode_steps = 10000
    env.rng_at_episode_start = np.random.RandomState()
    return env


class PadImage(gym.Wrapper):

    def step(self, action):
        # 64x63x3 -> 84x84x3
        obs, reward, done, info = self.unwrapped.step(action)
        obs = np.pad(obs, ((10, 10), (10, 10), (0, 0)))
        return obs, reward, done, info

    def reset(self):
        obs = self.unwrapped.reset()
        obs = np.pad(obs, ((10, 10), (10, 10), (0, 0)))
        return obs


class GrayScale(gym.Wrapper):

    def step(self, action):
        obs, reward, done, info = self.unwrapped.step(action)
        obs = obs.mean(-1)[..., None]
        return obs, reward, done, info

    def reset(self):
        obs = self.unwrapped.reset()
        obs = obs.mean(-1)[..., None]
        return obs


class TransposeImage(gym.Wrapper):

    def step(self, action):
        obs, reward, done, info = self.unwrapped.step(action)
        obs = obs.transpose((2, 0, 1))
        return obs, reward, done, info

    def reset(self):
        obs = self.unwrapped.reset()
        obs = obs.transpose((2, 0, 1))
        return obs


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # -> It's indeed batch normalization. :D
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
