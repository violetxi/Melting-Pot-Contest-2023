import os
import sys
# append path to this file to sys.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import meltingpot
from meltingpot import substrate
from ray import tune
from ray.rllib.policy import policy
from srl_dev.wrappers.downsamplesubstrate_wrapper import DownSamplingSubstrateWrapper
from srl_dev.wrappers.meltingpot_wrapper import MeltingPotEnv
import make_envs

SUPPORTED_SCENARIOS = [
    'allelopathic_harvest__open_0',
    'allelopathic_harvest__open_1',
    'allelopathic_harvest__open_2',
    'clean_up_2',
    'clean_up_3',
    'clean_up_4',
    'clean_up_5',
    'clean_up_6',
    'clean_up_7',
    'clean_up_8',
    'prisoners_dilemma_in_the_matrix__arena_0',
    'prisoners_dilemma_in_the_matrix__arena_1',
    'prisoners_dilemma_in_the_matrix__arena_2',
    'prisoners_dilemma_in_the_matrix__arena_3',
    'prisoners_dilemma_in_the_matrix__arena_4',
    'prisoners_dilemma_in_the_matrix__arena_5',
    'territory__rooms_0',
    'territory__rooms_1',
    'territory__rooms_2',
    'territory__rooms_3',
    # running with scissors arena
    'running_with_scissors_in_the_matrix__arena__rock_margin_0',
    'running_with_scissors_in_the_matrix__arena__rock_margin_1',
    'running_with_scissors_in_the_matrix__arena__paper_margin_0',
    'running_with_scissors_in_the_matrix__arena__paper_margin_1',
    'running_with_scissors_in_the_matrix__arena__scissors_margin_0',
    'running_with_scissors_in_the_matrix__arena__scissors_margin_1',
    # running with scissors repeated
    'running_with_scissors_in_the_matrix__repeated_0',
    'running_with_scissors_in_the_matrix__repeated_1',
    'running_with_scissors_in_the_matrix__repeated_2',
    'running_with_scissors_in_the_matrix__repeated_3',
    'running_with_scissors_in_the_matrix__repeated_4',
    'running_with_scissors_in_the_matrix__repeated_5',
    'running_with_scissors_in_the_matrix__repeated_6',
    'running_with_scissors_in_the_matrix__repeated_7',
    'running_with_scissors_in_the_matrix__repeated_8',
    # running with scissors one shot
    'running_with_scissors_in_the_matrix__one_shot_0',
    'running_with_scissors_in_the_matrix__one_shot_1',
    'running_with_scissors_in_the_matrix__one_shot_2',
    'running_with_scissors_in_the_matrix__one_shot_3',
]

IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']


def get_experiment_config(args, default_config):
    
    if args.exp == 'pd_arena':
        substrate_name = "prisoners_dilemma_in_the_matrix__arena"
    elif args.exp == 'al_harvest':
        substrate_name = "allelopathic_harvest__open"
    elif args.exp == 'clean_up':
        substrate_name = "clean_up"
    elif args.exp == 'territory_rooms':
        substrate_name = "territory__rooms"
    elif args.exp == 'rws_arena':
        substrate_name = "running_with_scissors_in_the_matrix__arena"
    elif args.exp == 'rws_repeated':
        substrate_name = "running_with_scissors_in_the_matrix__repeated"
    elif args.exp == 'rws_one_shot':
        substrate_name = "running_with_scissors_in_the_matrix__one_shot"
    else:
        raise Exception("Please set --exp to be one of ['pd_arena', 'al_harvest', 'clean_up', \
                        'territory_rooms', 'rws_arena', 'rws_repeated', 'rws_one_shot']. Other substrates are not supported.")

    # Fetch player roles
    player_roles = substrate.get_config(substrate_name).default_player_roles
    # use only 1 player
    player_roles = (player_roles[0],)

    if args.downsample:
        scale_factor = 8
    else:
        scale_factor = 1

    params_dict = {
        # resources
        "num_rollout_workers": args.num_workers,
        "num_gpus": args.num_gpus,

        # Env
        "env_name": "meltingpot",
        "env_config": {"substrate": substrate_name, "roles": player_roles, "scaled": scale_factor},

        # training
        # parameter explanation https://github.com/ray-project/ray/blob/c3a9756bf0c7691679edb679f666ae39614ba7e8/rllib/algorithms/algorithm_config.py#L1509        
        "seed": args.seed,
        "rollout_fragment_length": 1000,
        "train_batch_size": 10000,
        "sgd_minibatch_size": 2048,
        "disable_observation_precprocessing": True,
        "use_new_rl_modules": False,
        "use_new_learner_api": False,
        "framework": args.framework,

        # agent model
        "fcnet_hidden": (4, 4),
        "post_fcnet_hidden": (16,),
        "cnn_activation": "relu",
        "fcnet_activation": "relu",
        "post_fcnet_activation": "relu",
        "use_lstm": True,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": False,
        "lstm_cell_size": 2,
        "shared_policy": False,

        # experiment trials
        "exp_name": args.exp,
        "stopping": {
                    #"timesteps_total": 1000000,
                    "timesteps_total": 1e8,
                    #"training_iteration": 1000,
                    #"episode_reward_mean": 100,
        },
        # checkpoint config
        "num_checkpoints": 10,
        "checkpoint_interval": 5,
        "checkpoint_at_end": True,
        "checkpoint_score_attribute": "episode_reward_mean",
        "checkpoint_score_order": "max",
        # 
        "results_dir": args.results_dir,
        "logging": args.logging,

    }    
    # tunable parameters for ray.tune.Tuner
    # default_config contains default parameter for the RL algorithm (e.g. PPO)    
    run_configs = default_config
    """ Parameter tuning:  check what's inside the default_config to 
    identify what you want to tune """
    #run_configs['gamma'] = tune.uniform(0.9, 0.999)
    #run_configs['lr'] = tune.uniform(0.0001, 0.0005)
    #run_configs['entropy_coeff'] = tune.uniform(0.01, 0.3)
    # this is config for the tuner
    tune_configs = tune.TuneConfig(
      metric='episode_reward_mean',
      mode='max',
      num_samples=200,    # number of trials, -1 means infinite
      reuse_actors=False)  
    # TEMP
    tune_configs = None
    """ End Parameter tuning """
    # Resources 
    run_configs.num_rollout_workers = params_dict['num_rollout_workers']
    run_configs.num_gpus = params_dict['num_gpus']

    # Training
    run_configs.train_batch_size = params_dict['train_batch_size']
    run_configs.sgd_minibatch_size = params_dict['sgd_minibatch_size']
    run_configs.preprocessor_pref = None
    run_configs._disable_preprocessor_api = params_dict['disable_observation_precprocessing']
    run_configs.rl_module(_enable_rl_module_api=params_dict['use_new_rl_modules'])
    run_configs.training(_enable_learner_api=params_dict['use_new_learner_api'])
    run_configs = run_configs.framework(params_dict['framework'])
    run_configs.log_level = params_dict['logging']
    run_configs.seed = params_dict['seed']

    # Environment
    run_configs.env = params_dict['env_name']
    run_configs.env_config = params_dict['env_config']

    # Setup multi-agent policies. The below code will initialize independent
    # policies for each agent.
    #base_env = make_envs.env_creator(run_configs.env_config)
    # make environment with bots
    scenario = "running_with_scissors_in_the_matrix__repeated_8"
    factory = meltingpot.scenario.get_factory(scenario)
    base_env = factory.build()
    base_env = DownSamplingSubstrateWrapper(base_env, params_dict['env_config']['scaled'])
    base_env = MeltingPotEnv(base_env)

    policies = {}
    player_to_agent = {}
    for i in range(len(player_roles)):
        rgb_shape = base_env.observation_space[f"player_{i}"]["RGB"].shape
        sprite_x = rgb_shape[0]
        sprite_y = rgb_shape[1]

        policies[f"agent_{i}"] = policy.PolicySpec(
            observation_space=base_env.observation_space[f"player_{i}"],
            action_space=base_env.action_space[f"player_{i}"],
            config={
                "model": {
                    "conv_filters": [[16, [8, 8], 1],
                                    [128, [sprite_x, sprite_y], 1]],
                },
            })
        player_to_agent[f"player_{i}"] = f"agent_{i}"

    run_configs.multi_agent(policies=policies, policy_mapping_fn=(lambda agent_id, *args, **kwargs: 
                                                                  player_to_agent[agent_id]))    
    # Agent NN Model
    # Fully connect network with number of hidden layers to be used.
    run_configs.model["fcnet_hiddens"] = params_dict['fcnet_hidden']
    # Post conv fcnet with number of hidden layers to be used.
    run_configs.model["post_fcnet_hiddens"] = params_dict['post_fcnet_hidden']
    run_configs.model["conv_activation"] = params_dict['cnn_activation'] 
    run_configs.model["fcnet_activation"] = params_dict['fcnet_activation']
    run_configs.model["post_fcnet_activation"] = params_dict['post_fcnet_activation']
    run_configs.model["use_lstm"] = params_dict['use_lstm']
    run_configs.model["lstm_use_prev_action"] = params_dict['lstm_use_prev_action']
    run_configs.model["lstm_use_prev_reward"] = params_dict['lstm_use_prev_reward']
    run_configs.model["lstm_cell_size"] = params_dict['lstm_cell_size']
    """ Adding hyper-parameter to search """

    # ray air.RunConfig
    experiment_configs = {}
    experiment_configs['name'] = params_dict['exp_name']
    experiment_configs['stop'] = params_dict['stopping']
    # checkpoint config
    experiment_configs['keep'] = params_dict['num_checkpoints']
    experiment_configs['freq'] = params_dict['checkpoint_interval']
    experiment_configs['checkpoint_score_attr'] = params_dict['checkpoint_score_attribute']
    experiment_configs['checkpoint_score_order'] = params_dict['checkpoint_score_order']
    experiment_configs['end'] = params_dict['checkpoint_at_end']

    if args.framework == 'tf':
        experiment_configs['dir'] = f"{params_dict['results_dir']}/tf"
    else:
        experiment_configs['dir'] = f"{params_dict['results_dir']}/torch"
 
        
    return run_configs, experiment_configs, tune_configs