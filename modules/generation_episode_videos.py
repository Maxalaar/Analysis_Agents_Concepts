import gymnasium
import ray
from ray.rllib.algorithms import AlgorithmConfig

from utilities.path import PathManager
from utilities.ray.path_best_checkpoints import path_best_checkpoints
from utilities.ray.restore_best_algorithm import restore_best_algorithm
from utilities.video import generate_video


@ray.remote
class WorkerVideoEpisodeGeneration:
    def __init__(self, index, path_checkpoint, path_video, configuration: AlgorithmConfig):
        algorithm = configuration.algo_class(config=configuration)
        algorithm.restore(path_checkpoint)
        self.index = index
        self.environment = algorithm.env_creator(algorithm.config.env_config)
        self.policy = algorithm.get_policy()
        self.path_video = path_video

    def rollout(self, num_episodes):
        for index_episodes in range(num_episodes):
            renders = []

            observation, _ = self.environment.reset()
            terminated = False
            truncated = False
            while not terminated or truncated:
                observation = gymnasium.spaces.utils.flatten(self.environment.observation_space, observation)
                renders.append(self.environment.render())
                action, _, _ = self.policy.compute_single_action(observation, explore=True)
                observation, reward, terminated, truncated, information = self.environment.step(action)

            generate_video(renders, self.path_video + '/video_' + str(self.index) + '_' + str(index_episodes))


def generation_episode_videos(
        path_manager: PathManager,
        workers_number,
        number_episodes_per_worker,
):
    path_checkpoint = path_best_checkpoints(path_manager.rllib_trial_path)
    algorithm = restore_best_algorithm(path_manager.rllib_trial_path)
    configuration = algorithm.config.copy(copy_frozen=False)
    del algorithm
    configuration.learners(num_learners=0)
    configuration.env_runners(num_env_runners=0)

    workers = [WorkerVideoEpisodeGeneration.remote(worker_index, path_checkpoint, path_manager.episodes_directory, configuration) for worker_index in range(workers_number)]

    ray.get([worker.rollout.remote(number_episodes_per_worker) for worker in workers])
