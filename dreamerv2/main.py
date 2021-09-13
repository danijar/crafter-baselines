import gym
import crafter
import dreamerv2.api as dv2

config = dv2.configs.crafter.parse_flags()

env = gym.make('CrafterReward-v1')
env = crafter.Recorder(
    env, config.logdir,
    save_stats=True,
    save_video=False,
    save_episode=False,
)
env = dv2.GymWrapper(env)
env = dv2.OneHotAction(env)

dv2.train(env, config, [dv2.TerminalOutput()])
