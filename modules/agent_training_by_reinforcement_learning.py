from utilities.path import PathManager
from utilities.ray.reinforcement_learning import train


def agent_training_by_reinforcement_learning(
    path_manager: PathManager,
    environment_name,
    environment_configuration,
    architecture_name,
    architecture_configuration,
    stopping_criterion,
    train_batch_size=4000,
    num_learners=1,
    num_gpus_per_learner=0,
    num_cpus_per_learner=1,
    num_env_runners=5,
    evaluation_num_env_runners=1,
    evaluation_interval=100,
):
    train(
        rllib_trial_name=path_manager.rllib_trial_name,
        rllib_directory=path_manager.rllib_directory,
        environment_name=environment_name,
        environment_configuration=environment_configuration,
        architecture_name=architecture_name,
        architecture_configuration=architecture_configuration,
        stopping_criterion=stopping_criterion,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        num_cpus_per_learner=num_cpus_per_learner,
        num_env_runners=num_env_runners,
        evaluation_num_env_runners=evaluation_num_env_runners,
        evaluation_interval=evaluation_interval,
        train_batch_size=train_batch_size,
    )
