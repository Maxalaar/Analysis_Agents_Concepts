from utilities.path import PathManager
from utilities.ray.reinforcement_learning import train


def agent_training_by_reinforcement_learning(
    path_manager: PathManager,
    environment_name,
    environment_configuration,
    architecture_name,
    architecture_configuration,
    stopping_criterion,
):
    train(
        rllib_directory=path_manager.rllib_directory,
        environment_name=environment_name,
        environment_configuration=environment_configuration,
        architecture_name=architecture_name,
        architecture_configuration=architecture_configuration,
        stopping_criterion=stopping_criterion,
        num_learners=1,
        num_env_runners=10,
        evaluation_num_env_runners=1,
        evaluation_interval=100,
    )
