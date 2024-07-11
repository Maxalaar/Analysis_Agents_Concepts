from utilities.lightning import supervised_learning
from utilities.path import PathManager


def learning_concept_observation_correspondence(path_manager: PathManager, architecture):
    supervised_learning.train(
        data_path=path_manager.embeddings_dataset,
        architecture=architecture,
        tensorboard_path=path_manager.lightning_tensorboard_directory,
        model_path=path_manager.lightning_model_directory,
        x_name='embeddings_singular_basis',
        y_name='observations',
        accelerator='gpu',
        batch_size=64*64*64,
    )
