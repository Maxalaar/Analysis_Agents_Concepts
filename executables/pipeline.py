from datetime import timedelta

import ray

from architectures.reinforcement.register_architectures import register_architectures
from environments.register_environments import register_environments
from modules.agent_training_by_reinforcement_learning import agent_training_by_reinforcement_learning
from modules.generation_episode_videos import generation_episode_videos
from modules.generation_observation_dataset import generation_observation_dataset
from modules.generation_embeddings_dataset import generation_embeddings_dataset
from modules.concept_extraction.linear_perturbation_latent_space import linear_perturbation_latent_space
from modules.concept_extraction.pca_extraction_concepts import pca_extraction_concepts
from utilities.path import PathManager
from architectures.supervised.dense import Dense
import environments.environment_configurations as environment_configurations
from environments.pong_survivor.pong_survivor import PongSurvivor

if __name__ == '__main__':
    register_architectures()
    register_environments()

    ray.init(local_mode=False)
    experiment_name = 'debug_all_v3'
    path_manager = PathManager(experiment_name)

    # Environment
    environment_name = 'PongSurvivor'
    environment_creator = PongSurvivor
    environment_configuration = environment_configurations.classic_one_ball

    # Reinforcement Learning
    reinforcement_learning_architecture_name = 'dense'
    reinforcement_learning_architecture_configuration = {
        'configuration_hidden_layers': [128, 64, 32, 64, 128],
        'layers_use_clustering': [False, False, True, False, False],
    }
    stopping_criterion = {
        'time_total_s': 60 * 1 * 1,    # 20 minutes
        'env_runners/episode_reward_mean': 0.80,
    }
    train_batch_size = 16000
    num_learners = 1
    num_env_runners = 5
    evaluation_num_env_runners = 1
    evaluation_interval = 10

    # Generation dataset
    workers_number = 5
    number_episodes_per_worker = 1  # 50
    number_iterations = 1  # 50

    # Supervised Learning
    supervised_learning_architecture = Dense
    accelerator = 'gpu'
    max_time = timedelta(hours=0, minutes=1)   # 60 minutes
    batch_size = 64  # 64*64*64*10
    number_worker_datamodule = 4
    check_val_every_n_epoch = 1

    # Run
    agent_training_by_reinforcement_learning(
        path_manager=path_manager,
        environment_name=environment_name,
        environment_configuration=environment_configuration,
        architecture_name=reinforcement_learning_architecture_name,
        architecture_configuration=reinforcement_learning_architecture_configuration,
        train_batch_size=train_batch_size,
        stopping_criterion=stopping_criterion,
        num_learners=num_learners,
        num_env_runners=num_env_runners,
        evaluation_num_env_runners=evaluation_num_env_runners,
        evaluation_interval=evaluation_interval,
    )

    generation_episode_videos(
        path_manager=path_manager,
        workers_number=workers_number,
        number_episodes_per_worker=2,
    )

    generation_observation_dataset(
        path_manager=path_manager,
        workers_number=workers_number,
        number_episodes_per_worker=number_episodes_per_worker,
        number_iterations=number_iterations,
    )

    generation_embeddings_dataset(
        path_manager=path_manager,
        workers_number=2,
    )

    pca_extraction_concepts(
        path_manager=path_manager,
        architecture=supervised_learning_architecture,
        batch_size=batch_size,
        max_time=max_time,
        environment_creator=environment_creator,
        environment_configuration=environment_configuration,
        decomposition=True,
        learning=True,
        generation=True,
        accelerator=accelerator,
        number_worker_datamodule=number_worker_datamodule,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    linear_perturbation_latent_space(
        path_manager=path_manager,
        architecture=supervised_learning_architecture,
        perturbations_number=10,
        perturbation_magnitude=6,
        batch_size=batch_size,
        max_time=max_time,
        environment_creator=environment_creator,
        environment_configuration=environment_configuration,
        accelerator=accelerator,
        number_worker_datamodule=number_worker_datamodule,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
