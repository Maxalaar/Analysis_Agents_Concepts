import os

from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN


def train(
    rllib_trial_name,
    rllib_directory,
    environment_name: str,
    environment_configuration: dict,
    architecture_name: str,
    architecture_configuration: dict,
    stopping_criterion: dict,
    train_batch_size: int = 4000,
    num_env_runners: int = 0,
    num_learners: int = 0,
    evaluation_num_env_runners: int = 0,
    num_envs_per_env_runner: int = 1,
    checkpoint_frequency: int = 100,
    num_cpus_per_env_runner: int = 1,
    num_gpus_per_env_runner: float = 0,
    num_cpus_per_learner: int = 1,
    num_gpus_per_learner: float = 0,
    evaluation_interval: int = 0,
    evaluation_duration: int = 100,
):
    algorithm_configuration: AlgorithmConfig = (
        PPOConfig()
        .environment(env=environment_name, env_config=environment_configuration)
        .framework('torch')
        # .resources(
        #     num_gpus=0,
        # )
        .training(
            model={
                'custom_model': architecture_name,
                'custom_model_config': architecture_configuration,
            },
            train_batch_size=4000 * 2,
            # mini_batch_size_per_learner=4000 * 5,    # 128 * 2
            # sgd_minibatch_size=4000 * 5,
            # num_sgd_iter=30 * 4,
            # lr=1e-5,
        )
        .env_runners(
            batch_mode='complete_episodes',
            # exploration_config={
            #     'type': 'StochasticSampling',
            # },
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            num_gpus_per_env_runner=num_gpus_per_env_runner,
        )
        .learners(
            num_learners=num_learners,
            num_cpus_per_learner=num_cpus_per_learner,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        .evaluation(
            evaluation_interval=evaluation_interval,
            evaluation_num_env_runners=evaluation_num_env_runners,
            evaluation_duration=evaluation_duration,
            evaluation_parallel_to_training=True,
        )
    )

    tuner = tune.Tuner(
        trainable=PPO,
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            name=rllib_trial_name,
            storage_path=rllib_directory,
            stop=stopping_criterion,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute='env_runners/episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=checkpoint_frequency,
                checkpoint_at_end=True,
            )
        ),
    )

    tuner.fit()
