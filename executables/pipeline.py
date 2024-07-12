from datetime import timedelta

import ray

from architectures.reinforcement.register_architectures import register_architectures
from environments.register_environments import register_environments
from modules.agent_training_by_reinforcement_learning import agent_training_by_reinforcement_learning
from modules.generation_observation_dataset import generation_observation_dataset
from modules.generation_observations_based_concepts import generation_observations_based_concepts
from modules.generation_projection_dataset import generation_projection_dataset
from modules.learning_concept_observation_correspondence import learning_concept_observation_correspondence
from modules.singular_value_decomposition_latent_spaces import singular_value_decomposition_embeddings
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
        'time_total_s': 60 * 60,
        'env_runners/episode_reward_mean': 0.95,
    }

    # Generation dataset
    workers_number = 10
    number_episodes_per_worker = 100
    number_iterations = 10

    # Supervised Learning
    supervised_learning_architecture = Dense
    max_time = timedelta(minutes=60)
    batch_size = 64*64*64

    # Run
    agent_training_by_reinforcement_learning(
        path_manager=path_manager,
        environment_name=environment_name,
        environment_configuration=environment_configuration,
        architecture_name=reinforcement_learning_architecture_name,
        architecture_configuration=reinforcement_learning_architecture_configuration,
        stopping_criterion=stopping_criterion,
    )

    generation_observation_dataset(
        path_manager=path_manager,
        workers_number=workers_number,
        number_episodes_per_worker=number_episodes_per_worker,
        number_iterations=number_iterations,
    )

    generation_projection_dataset(
        path_manager=path_manager,
        workers_number=2,
    )

    singular_value_decomposition_embeddings(
        path_manager=path_manager
    )

    learning_concept_observation_correspondence(
        path_manager=path_manager,
        architecture=supervised_learning_architecture,
        batch_size=batch_size,
        max_time=max_time,
    )

    environment = PongSurvivor(environment_configuration)
    environment.render_mode = 'rgb_array'
    generation_observations_based_concepts(
        path_manager=path_manager,
        environment=environment,
    )



    
