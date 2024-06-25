import os


class PathManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.execution_directory = os.getcwd()
        self.results_directory = os.path.join(self.execution_directory, 'results')
        self.experiment_directory = os.path.join(self.results_directory, self.experiment_name)

        self.datasets_directory = os.path.join(self.experiment_directory, 'datasets')
        self.lightning_directory = os.path.join(self.experiment_directory, 'lightning')
