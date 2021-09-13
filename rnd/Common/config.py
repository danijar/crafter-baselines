import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--logdir", default='logdir')
    parser.add_argument("--reward", default=True, type=lambda x: bool(['false', 'true'].index(x.lower())))
    # TODO: Should be 1 but only works for 2
    parser.add_argument("--n_workers", default=1, type=int, help="Number of parallel environments.")
    parser.add_argument("--interval", default=50, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by iterations.")
    parser.add_argument("--do_test", action="store_false",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--render", action="store_true",
                        help="The flag determines whether to render each agent or not.")
    parser.add_argument("--train_from_scratch", action="store_false",
                        help="The flag determines whether to train from scratch or continue previous tries.")

    parser_params = parser.parse_args()

    """
     Parameters based on the "Exploration By Random Network Distillation" paper.
     https://arxiv.org/abs/1810.12894
    """

    if parser_params.reward:
      env_name = "CrafterReward-v1"
    else:
      env_name = "CrafterNoReward-v1"

    # region default parameters
    default_params = {"env_name": env_name,
                      "state_shape": (4, 84, 84),
                      "obs_shape": (1, 84, 84),
                      "total_rollouts_per_env": int(5e6),
                      "max_frames_per_episode": 10000,
                      "rollout_length": 128,
                      "n_epochs": 4,
                      "n_mini_batch": 4,
                      "lr": 1e-4,
                      "ext_gamma": 0.999,
                      "int_gamma": 0.99,
                      "lambda": 0.95,
                      "ext_adv_coeff": 2 if parser_params.reward else 0,
                      "int_adv_coeff": 1,
                      "ent_coeff": 0.001,
                      "clip_range": 0.1,
                      "pre_normalization_steps": 50,
                      }

    # endregion
    total_params = default_params.copy()
    total_params.update(**vars(parser_params))
    print("params:", total_params)
    return total_params
