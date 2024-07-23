from datetime import timedelta

import ray

from architectures.reinforcement.register_architectures import register_architectures
from environments.register_environments import register_environments
from modules.agent_training_by_reinforcement_learning import agent_training_by_reinforcement_learning
from modules.generation_observation_dataset import generation_observation_dataset
from modules.generation_embeddings_dataset import generation_embeddings_dataset
from modules.methods_concept_extraction.pca_extraction_concepts import pca_extraction_concepts
from utilities.path import PathManager
from architectures.supervised.dense import Dense
import environments.environment_configurations as environment_configurations
from environments.pong_survivor.pong_survivor import PongSurvivor


if __name__ == '__main__':
    register_architectures()
    register_environments()

    ray.init(local_mode=False)
    experiment_name = 'debug_all'
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
        'time_total_s': 60 * 60 * 2,
        'env_runners/episode_reward_mean': 0.80,
    }
    num_learners = 1
    num_env_runners = 5
    evaluation_num_env_runners = 1
    evaluation_interval = 10

    # Generation dataset
    workers_number = 5
    number_episodes_per_worker = 50
    number_iterations = 50

    # Supervised Learning
    supervised_learning_architecture = Dense
    accelerator = 'gpu'
    max_time = timedelta(hours=2, minutes=0)
    batch_size = 64*64*64*10
    number_worker_datamodule = 4
    check_val_every_n_epoch = 100


    # Run
    agent_training_by_reinforcement_learning(
        path_manager=path_manager,
        environment_name=environment_name,
        environment_configuration=environment_configuration,
        architecture_name=reinforcement_learning_architecture_name,
        architecture_configuration=reinforcement_learning_architecture_configuration,
        stopping_criterion=stopping_criterion,
        num_learners=num_learners,
        num_env_runners=num_env_runners,
        evaluation_num_env_runners=evaluation_num_env_runners,
        evaluation_interval=evaluation_interval,
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






    
