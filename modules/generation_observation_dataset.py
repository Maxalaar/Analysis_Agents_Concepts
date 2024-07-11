import gymnasium
import numpy as np
import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.tune import Tuner

from utilities.data import save_data_to_h5
from utilities.path import PathManager


@ray.remote
class WorkerObservationGeneration:
    def __init__(self, path_checkpoint, configuration: AlgorithmConfig):
        algorithm = configuration.algo_class(config=configuration)
        algorithm.restore(path_checkpoint)
        self.environment = algorithm.env_creator(algorithm.config.env_config)
        self.policy = algorithm.get_policy()
        self.observations = None

    def rollout(self, num_episodes):
        self.observations = []
        for _ in range(num_episodes):
            observation, _ = self.environment.reset()
            terminated = False
            truncated = False
            while not terminated or truncated :
                observation = gymnasium.spaces.utils.flatten(self.environment.observation_space, observation)
                self.observations.append(observation)
                action, _, _ = self.policy.compute_single_action(observation, explore=True)
                observation, reward, terminated, truncated, information = self.environment.step(action)

    def save(self, path):
        save_data_to_h5(
            filename=path,
            dataset_name='observations',
            data=np.array(self.observations),
        )
        del self.observations


def generation_observation_dataset(
        path_manager: PathManager,
        workers_number,
        number_episodes_per_worker,
        number_iterations,
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

    workers = [WorkerObservationGeneration.remote(path_checkpoint, configuration) for _ in range(workers_number)]

    for i in range(number_iterations):
        ray.get([worker.rollout.remote(number_episodes_per_worker) for worker in workers])
        for worker in workers:
            ray.get(worker.save.remote(path_manager.observations_dataset))
