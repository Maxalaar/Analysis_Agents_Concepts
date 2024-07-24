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
        self.rllib_trial_name = 'rllib_base_trial'
        self.rllib_trial_path = os.path.join(self.rllib_directory, self.rllib_trial_name)

        self.datasets_directory = os.path.join(self.experiment_directory, 'datasets')
        self.observations_dataset = os.path.join(self.datasets_directory, 'observations.h5')
        self.embeddings_dataset = os.path.join(self.datasets_directory, 'embeddings.h5')

        self.lightning_directory = os.path.join(self.experiment_directory, 'lightning')
        self.lightning_tensorboard_directory = os.path.join(self.lightning_directory, 'tensorboard')
        self.lightning_models_directory = os.path.join(self.lightning_directory, 'model')

        self.images_directory = os.path.join(self.experiment_directory, 'images')
        self.concepts_directory = os.path.join(self.images_directory, 'concepts')

        self.videos_directory = os.path.join(self.experiment_directory, 'videos')
        self.episodes_directory = os.path.join(self.videos_directory, 'episodes')
