from ray.tune import Tuner


def path_best_checkpoints(rllib_trial_path, trainable='PPO'):
    tuner = Tuner.restore(path=rllib_trial_path, trainable=trainable)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='evaluation/env_runners/episode_reward_mean', mode='max')
    path_checkpoint = best_result.best_checkpoints[0][0].path
    return path_checkpoint
