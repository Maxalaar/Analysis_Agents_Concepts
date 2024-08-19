from datetime import timedelta

from architectures.supervised.variational_autoencoder import VariationalAutoencoder
from utilities.concept_extraction.comparing_generations import comparing_generations
from utilities.data import DataModule
from utilities.lightning.autoencoding import Autoencoding
from utilities.lightning.load import load
from utilities.path import PathManager


def pretraining_variational_autoencoder(path_manager: PathManager, environment_creator, environment_configuration):
    model_name = 'pretraining_variational_autoencoder'
    data_observations = DataModule(
        path='/mnt/873a422e-1bf1-4c0b-b123-e47e6872b379/Programming_Projects/Pycharm_Projects/Analysis_Agents_Concepts/experiments/dense/datasets/observations.h5',
        x_name='observations',
        batch_size=2**14,
        number_workers=5,
    )
    autoencoding = Autoencoding(
        name=model_name,
        architecture=VariationalAutoencoder,
        input_shape=data_observations.x_shape,
        output_shape=data_observations.x_shape,
        save_path=path_manager.lightning_models_directory,
        tensorboard_path=path_manager.lightning_tensorboard_directory,
    )
    autoencoding.learning(
        data=data_observations,
        accelerator='gpu',
        max_time=timedelta(hours=24, minutes=0),
    )
    comparing_generations(
        path_manager=path_manager,
        data_path='/mnt/873a422e-1bf1-4c0b-b123-e47e6872b379/Programming_Projects/Pycharm_Projects/Analysis_Agents_Concepts/experiments/dense/datasets/observations.h5',
        x_name='observations',
        y_name='observations',
        model=load(model_directory=path_manager.lightning_models_directory + '/' + model_name,
                   learning_type=Autoencoding),
        sample_size=1000,
        environment=environment_creator(environment_configuration),
    )


