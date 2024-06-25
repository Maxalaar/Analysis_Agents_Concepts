from utilities.lightning import learning
from utilities.path import PathManager


def learning_concept_observation_correspondence(path_manager: PathManager, architecture):
    learning.train(
        data_path=path_manager.latent_space_singular_basis,
        architecture=architecture,
        tensorboard_path=path_manager.lightning_tensorboard_directory,
        model_path=path_manager.lightning_model_directory,
        x_name='latent_space_singular_basis',
        y_name='observation',
        accelerator='gpu',
    )
