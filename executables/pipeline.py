import ray

from architectures.reinforcement.register_architectures import register_architectures
from environments.register_environments import register_environments
from modules.agent_training_by_reinforcement_learning import agent_training_by_reinforcement_learning
from modules.generation_observations_based_concepts import generation_observations_based_concepts
from modules.learning_concept_observation_correspondence import learning_concept_observation_correspondence
from modules.singular_value_decomposition_latent_spaces import singular_value_decomposition_embeddings
from utilities.path import PathManager
from architectures.supervised.simple import Simple
import environments.environment_configurations as environment_configurations
from environments.pong_survivor.pong_survivor import PongSurvivor


if __name__ == '__main__':
    register_architectures()
    register_environments()

    ray.init(local_mode=False)
    experiment_name = 'debug_rl'
    path_manager = PathManager(experiment_name)

    # Environment
    environment_name = 'PongSurvivor'
    environment_configuration = environment_configurations.classic_one_ball

    # Reinforcement Learning
    reinforcement_learning_architecture_name = 'dense'
    reinforcement_learning_architecture_configuration = {
        'configuration_hidden_layers': [128, 64, 32, 64, 128],
        'layers_use_clustering': [False, False, True, False, False],
    }
    stopping_criterion = {
        'time_total_s': 60 * 60 * 1,
        'env_runners/episode_reward_mean': 0.95,
    }

    # Supervised Learning
    supervised_learning_architecture = Simple

    # Run
    agent_training_by_reinforcement_learning(
        path_manager=path_manager,
        environment_name=environment_name,
        environment_configuration=environment_configuration,
        architecture_name=reinforcement_learning_architecture_name,
        architecture_configuration=reinforcement_learning_architecture_configuration,
        stopping_criterion=stopping_criterion,
    )

    # singular_value_decomposition_embeddings(
    #     path_manager=path_manager
    # )
    #
    # learning_concept_observation_correspondence(
    #     path_manager=path_manager,
    #     architecture=supervised_learning_architecture,
    # )
    #
    # environment = PongSurvivor(environment_configuration)
    # environment.render_mode = 'rgb_array'
    # generation_observations_based_concepts(
    #     path_manager=path_manager,
    #     environment=environment,
    # )



    
