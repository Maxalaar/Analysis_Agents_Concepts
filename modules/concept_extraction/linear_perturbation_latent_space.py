from datetime import timedelta

from utilities.concept_extraction.comparing_generations import comparing_generations
from utilities.concept_extraction.generation_observations import generation_observations
from utilities.concept_extraction.variation_observation_according_concepts import variation_observation_according_concepts
from utilities.data import DataModule
from utilities.lightning.perturbation import Perturbation
from utilities.lightning.supervised import Supervised
from utilities.path import PathManager
from utilities.lightning.load import load


def linear_perturbation_latent_space(
    path_manager: PathManager,
    architecture,
    batch_size,
    max_time,
    perturbations_number,
    perturbation_magnitude,
    environment_creator,
    environment_configuration,
    accelerator='gpu',
    number_worker_datamodule=1,
    check_val_every_n_epoch=1,
):
    data_embeddings_to_observations = DataModule(
        path=path_manager.embeddings_dataset,
        x_name='embeddings',
        y_name='observations',
        batch_size=batch_size,
        number_workers=number_worker_datamodule
    )
    embeddings_to_observations = Supervised(
        name='embeddings_to_observations',
        architecture=architecture,
        input_shape=data_embeddings_to_observations.x_shape,
        output_shape=data_embeddings_to_observations.y_shape,
        save_path=path_manager.lightning_models_directory,
        tensorboard_path=path_manager.lightning_tensorboard_directory,
    )
    embeddings_to_observations.learning(
        data=data_embeddings_to_observations,
        max_time=max_time,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
    )
    del data_embeddings_to_observations

    comparing_generations(
        path_manager=path_manager,
        data_path=path_manager.embeddings_dataset,
        x_name='embeddings',
        y_name='observations',
        model=load(model_directory=path_manager.lightning_models_directory + '/' + str('embeddings_to_observations'), learning_type=Supervised),
        sample_size=1000,
        environment=environment_creator(environment_configuration),
    )

    generation_observations(
        path_manager=path_manager,
        model=load(model_directory=path_manager.lightning_models_directory + '/' + str('embeddings_to_observations'), learning_type=Supervised),
        sample_size=1000,
        environment=environment_creator(environment_configuration),
    )

    embeddings_to_observations = load(
        model_directory=path_manager.lightning_models_directory + '/' + str('embeddings_to_observations'),
        learning_type=Supervised,
    )
    data_perturbator = DataModule(
        path=path_manager.embeddings_dataset,
        x_name='embeddings',
        batch_size=16_384,
        number_workers=number_worker_datamodule
    )
    perturbation = Perturbation(
        name='perturbation',
        architecture=architecture,
        observation_shape=embeddings_to_observations.model.output_dimension,
        embeddings_to_observations=embeddings_to_observations,
        perturbations_number=perturbations_number,
        embedding_shape=embeddings_to_observations.model.input_dimension,
        perturbation_magnitude=perturbation_magnitude,
        save_path=path_manager.lightning_models_directory,
        tensorboard_path=path_manager.lightning_tensorboard_directory,
        layer_configuration=[1024, 512, 256, 128, 64],
    )
    perturbation.learning(
        data=data_perturbator,
        max_time=timedelta(days=7, hours=0, minutes=0),
        patience=10e10,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
    )

    variation_observation_according_concepts(
        path_manager,
        path_manager.embeddings_dataset,
        'embeddings',
        'perturbation',
        'embeddings_to_observations',
        environment_creator(environment_configuration),
    )





