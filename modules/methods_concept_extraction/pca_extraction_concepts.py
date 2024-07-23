from utilities.concept_extraction.generation_observations_based_concepts import generation_observations_based_concepts
from utilities.concept_extraction.learning_concept_observation_correspondence import learning_concept_observation_correspondence
from utilities.path import PathManager
from utilities.concept_extraction.principal_component_analysis_embeddings import principal_component_analysis_embeddings


def pca_extraction_concepts(
    path_manager: PathManager,
    architecture,
    batch_size,
    max_time,
    environment_creator,
    environment_configuration,
    decomposition=True,
    learning=True,
    generation=True,
    accelerator='gpu',
    number_worker_datamodule=1,
    check_val_every_n_epoch=1,
):
    if decomposition:
        principal_component_analysis_embeddings(
            path_manager=path_manager,
        )

    if learning:
        learning_concept_observation_correspondence(
            model_name='pca_to_obs',
            path_manager=path_manager,
            architecture=architecture,
            batch_size=batch_size,
            max_time=max_time,
            x_name='embeddings_principal_component_bases',
            accelerator=accelerator,
            number_worker_datamodule=number_worker_datamodule,
            check_val_every_n_epoch=check_val_every_n_epoch,
        )

    if generation:
        environment = environment_creator(environment_configuration)
        environment.render_mode = 'rgb_array'
        generation_observations_based_concepts(
            path_manager=path_manager,
            model_name='pca_to_obs',
            environment=environment,
        )
