import numpy as np
import ray
import h5py
from ray.rllib.algorithms import Algorithm
from ray.tune import Tuner
import torch

from utilities.data import save_data_to_h5
from utilities.path import PathManager


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


def generation_projection_dataset(
        path_manager: PathManager,
        workers_number,
):
    tuner = Tuner.restore(path=path_manager.rllib_trial_path, trainable='PPO')
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint = best_result.best_checkpoints[0][0].path

    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    configuration = algorithm.config.copy(copy_frozen=False)
    del algorithm
    configuration.learners(num_learners=0)
    configuration.env_runners(num_env_runners=0)

    workers = [WorkerProjectionGeneration.remote(path_checkpoint, configuration) for _ in range(workers_number)]

    with h5py.File(path_manager.observations_dataset, 'r') as file:
        observations = file['observations']
        observations = np.array_split(observations, workers_number)

        worker_tasks = ray.get([worker.projections.remote(observations[index]) for index, worker in enumerate(workers)])
        for worker in workers:
            ray.get(worker.save.remote(path_manager.embeddings_dataset))

        save_data_to_h5(
            filename=path_manager.embeddings_dataset,
            dataset_name='observations',
            data=file['observations'],
        )
