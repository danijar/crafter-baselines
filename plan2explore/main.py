import gym
import crafter
import dreamerv2.api as dv2

config = dv2.defaults
config = config.update(dv2.configs['crafter'])
config = config.update({
    'expl_behavior': 'Plan2Explore',
    'pred_discount': False,
    'grad_heads': ['decoder'],
    'expl_intr_scale': 1.0,
    'expl_extr_scale': 0.0,
    'discount': 0.99,
})
config = config.parse_flags()

env = gym.make('CrafterNoReward-v1')
env = crafter.Recorder(
    env, config.logdir,
    save_stats=True,
    save_video=False,
    save_episode=False,
)

dv2.train(env, config, [dv2.TerminalOutput()])
