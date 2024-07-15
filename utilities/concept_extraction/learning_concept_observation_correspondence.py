from utilities.lightning import supervised_learning
from utilities.path import PathManager


def learning_concept_observation_correspondence(path_manager: PathManager, architecture, model_name, batch_size, max_time, x_name):
    supervised_learning.train(
        model_name=model_name,
        data_path=path_manager.embeddings_dataset,
        architecture=architecture,
        tensorboard_path=path_manager.lightning_tensorboard_directory,
        model_path=path_manager.lightning_models_directory,
        x_name=x_name,
        y_name='observations',
        accelerator='gpu',
        batch_size=batch_size,
        max_time=max_time,
    )
