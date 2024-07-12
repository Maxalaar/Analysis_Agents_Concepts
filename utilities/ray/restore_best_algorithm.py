from ray.rllib.algorithms import Algorithm

from utilities.ray.path_best_checkpoints import path_best_checkpoints


def restore_best_algorithm(rllib_trial_path):
    path_checkpoint = path_best_checkpoints(rllib_trial_path)
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    return algorithm
