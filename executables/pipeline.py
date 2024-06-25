from modules import supervised_learning
from utilities.path import PathManager
from architectures.supervised.simple import Simple

if __name__ == '__main__':
    experiment_name = 'debug'
    path_manager = PathManager(experiment_name)

    supervised_learning_architecture = Simple

    supervised_learning.train(
        experiment_name=path_manager.experiment_name,
        data_path=path_manager.datasets_directory + '/latent_space_singular_basis.h5',
        x_name='latent_space_singular_basis',
        y_name='observation',
        architecture=supervised_learning_architecture,
        save_path=path_manager.lightning_directory,
        accelerator='gpu',
    )

    
