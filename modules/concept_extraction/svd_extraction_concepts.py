from utilities.concept_extraction.generation_observations_based_concepts import generation_observations_based_concepts
from utilities.data import DataModule
from utilities.lightning.supervised import Supervised
from utilities.path import PathManager
from utilities.concept_extraction.singular_value_decomposition_embeddings import singular_value_decomposition_embeddings


def svd_extraction_concepts(
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
    model_name = 'svd_to_observations'
    if decomposition:
        singular_value_decomposition_embeddings(
            path_manager=path_manager,
        )

    if learning:
        data = DataModule(
            path=path_manager.embeddings_dataset,
            x_name='embeddings_singular_basis',
            y_name='observations',
            batch_size=batch_size,
            number_workers=number_worker_datamodule
        )
        svd_to_observations = Supervised(
            name='svd_to_observations',
            architecture=architecture,
            input_shape=data.x_shape,
            output_shape=data.y_shape,
            save_path=path_manager.lightning_models_directory,
            tensorboard_path=path_manager.lightning_tensorboard_directory,
        )
        svd_to_observations.learning(
            data=data,
            max_time=max_time,
            check_val_every_n_epoch=check_val_every_n_epoch,
            accelerator=accelerator,
        )

    if generation:
        environment = environment_creator(environment_configuration)
        generation_observations_based_concepts(
            path_manager=path_manager,
            model_name=model_name,
            environment=environment,
        )
