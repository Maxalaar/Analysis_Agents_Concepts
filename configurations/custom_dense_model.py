from architectures.supervised.dense import Dense
from utilities.configuration import ExperimentConfiguration, ReinforcementLearningConfiguration, \
    LinearPerturbationLatentSpaceConfiguration

reinforcement_learning_configuration = ReinforcementLearningConfiguration(
    architecture_name='dense',
    architecture_configuration={
        'configuration_hidden_layers': [128, 64, 32, 64, 128],
        'layers_use_clustering': [False, False, True, False, False],
    },
    stopping_criterion={
        'time_total_s': 60 * 20 * 1,
        'env_runners/episode_reward_mean': 0.85,
    },
)

linear_perturbation_latent_space_configuration = LinearPerturbationLatentSpaceConfiguration(
    architecture=Dense,
)

custom_dense_model = ExperimentConfiguration(
    experiment_name='custom_dense_model',
    reinforcement_learning_configuration=reinforcement_learning_configuration,
)
