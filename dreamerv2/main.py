import gym
import crafter
import dreamerv2.api as dv2

config = dv2.defaults
config = config.update(dv2.configs['crafter'])
config = config.parse_flags()

env = gym.make('CrafterReward-v1')
env = crafter.Recorder(
    env, config.logdir,
    save_stats=True,
    save_video=False,
    save_episode=False,
)

dv2.train(env, config, [dv2.TerminalOutput()])
