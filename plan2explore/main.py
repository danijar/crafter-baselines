import gym
import crafter
import dreamerv2.api as dv2

config = dv2.configs.crafter.update({
    'expl_behavior': 'Plan2Explore',
    'pred_discount': False,
    'grad_heads': ['decoder'],
    'expl_intr_scale': 1.0,
    'expl_extr_scale': 0.0,
    'discount': 0.99,
}).parse_flags()

env = gym.make('CrafterNoReward-v1')
env = crafter.Recorder(
    env, config.logdir,
    save_stats=True,
    save_video=False,
    save_episode=False,
)
env = dv2.GymWrapper(env)
env = dv2.OneHotAction(env)

dv2.train(env, config, [dv2.TerminalOutput()])
