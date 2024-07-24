from utilities.concept_extraction.generation_observations_based_concepts import generation_observations_based_concepts
import utilities.lightning.learning
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
        utilities.lightning.learning.train(
            model_name='pca_to_obs',
            data_path=path_manager.embeddings_dataset,
            architecture=architecture,
            tensorboard_path=path_manager.lightning_tensorboard_directory,
            model_path=path_manager.lightning_models_directory,
            x_name='embeddings_principal_component_bases',
            y_name='observations',
            accelerator=accelerator,
            batch_size=batch_size,
            max_time=max_time,
            number_worker_datamodule=number_worker_datamodule,
            check_val_every_n_epoch=check_val_every_n_epoch,
        )

    if generation:
        environment = environment_creator(environment_configuration)
        generation_observations_based_concepts(
            path_manager=path_manager,
            model_name='pca_to_obs',
            environment=environment,
        )
