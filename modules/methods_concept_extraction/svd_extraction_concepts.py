from utilities.concept_extraction.generation_observations_based_concepts import generation_observations_based_concepts
from utilities.concept_extraction.learning_concept_observation_correspondence import learning_concept_observation_correspondence
from utilities.path import PathManager
from utilities.concept_extraction.singular_value_decomposition_embeddings import singular_value_decomposition_embeddings


def svd_extraction_concepts(
    path_manager: PathManager,
    architecture,
    batch_size,
    max_time,
    environment_creator,
    environment_configuration
):
    singular_value_decomposition_embeddings(
        path_manager=path_manager,
    )

    learning_concept_observation_correspondence(
        path_manager=path_manager,
        model_name='svd_to_obs',
        architecture=architecture,
        batch_size=batch_size,
        max_time=max_time,
        x_name='embeddings_singular_basis',
    )

    environment = environment_creator(environment_configuration)
    environment.render_mode = 'rgb_array'
    generation_observations_based_concepts(
        path_manager=path_manager,
        model_name='svd_to_obs',
        environment=environment,
    )
