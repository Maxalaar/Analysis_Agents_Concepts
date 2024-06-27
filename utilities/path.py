import os
import glob


def find_latest_file(directory):
    files = glob.glob(os.path.join(directory, '*'))

    if not files:
        raise FileNotFoundError(f'No files found in directory: {directory}')

    latest_file = max(files, key=os.path.getmtime)

    return latest_file


class PathManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.execution_directory = os.getcwd()
        self.experiments_directory = os.path.join(self.execution_directory, 'experiments')
        self.experiment_directory = os.path.join(self.experiments_directory, self.experiment_name)

        self.rllib_directory = os.path.join(self.experiment_directory, 'rllib')

        self.datasets_directory = os.path.join(self.experiment_directory, 'datasets')
        self.embeddings_dataset = os.path.join(self.datasets_directory, 'embeddings.h5')

        self.lightning_directory = os.path.join(self.experiment_directory, 'lightning')
        self.lightning_tensorboard_directory = os.path.join(self.lightning_directory, 'tensorboard')
        self.lightning_model_directory = os.path.join(self.lightning_directory, 'model')

        self.images_directory = os.path.join(self.experiment_directory, 'images')
        self.concepts_directory = os.path.join(self.images_directory, 'concepts')
