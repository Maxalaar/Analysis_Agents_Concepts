from modules.generation_observations_based_concepts import generation_observations_based_concepts
from modules.learning_concept_observation_correspondence import learning_concept_observation_correspondence
from utilities.path import PathManager
from architectures.supervised.simple import Simple
import environments.environment_configurations as environment_configurations
from environments.pong_survivor.pong_survivor import PongSurvivor

if __name__ == '__main__':
    experiment_name = 'debug'
    path_manager = PathManager(experiment_name)
    environment_name = 'PongSurvivor'
    environment_configuration = environment_configurations.classic_one_ball

    supervised_learning_architecture = Simple

    # learning_concept_observation_correspondence(
    #     path_manager=path_manager,
    #     architecture=supervised_learning_architecture,
    # )

    generation_observations_based_concepts(
        path_manager=path_manager,
        environment=PongSurvivor(environment_configuration),
    )



    
