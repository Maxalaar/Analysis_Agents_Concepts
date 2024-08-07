import numpy as np
import ray
import h5py
import torch

from utilities.data import save_data_to_h5
from utilities.path import PathManager
from utilities.ray.path_best_checkpoints import path_best_checkpoints
from utilities.ray.restore_best_algorithm import restore_best_algorithm


@ray.remote
class WorkerProjectionGeneration:
    def __init__(self, path_checkpoint, configuration):
        algorithm = configuration.algo_class(config=configuration)
        algorithm.restore(path_checkpoint)
        self.policy = algorithm.get_policy()
        self.projections = None

    def projections(self, observations):
        self.projections = self.policy.model.get_latent_space(torch.tensor(observations).to(self.policy.device))

    def save(self, path):
        save_data_to_h5(
            filename=path,
            dataset_name='embeddings',
            data=self.projections,
        )
        del self.projections


def generation_embeddings_dataset(
        path_manager: PathManager,
        workers_number,
):
    path_checkpoint = path_best_checkpoints(path_manager.rllib_trial_path)
    algorithm = restore_best_algorithm(path_manager.rllib_trial_path)
    configuration = algorithm.config.copy(copy_frozen=False)
    del algorithm
    configuration.learners(num_learners=0)
    configuration.env_runners(num_env_runners=0)

    workers = [WorkerProjectionGeneration.remote(path_checkpoint, configuration) for _ in range(workers_number)]

    with h5py.File(path_manager.observations_dataset, 'r') as file:
        observations = file['observations']
        observations = np.array_split(observations, workers_number)

        ray.get([worker.projections.remote(observations[index]) for index, worker in enumerate(workers)])
        for worker in workers:
            ray.get(worker.save.remote(path_manager.embeddings_dataset))

        save_data_to_h5(
            filename=path_manager.embeddings_dataset,
            dataset_name='observations',
            data=file['observations'],
        )
