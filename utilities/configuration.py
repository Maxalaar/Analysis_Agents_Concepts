from architectures.supervised.dense import Dense
from environments.pong_survivor.pong_survivor import PongSurvivor
import environments.configurations as environment_configurations


class EnvironmentConfiguration:
    def __init__(
            self,
            name='PongSurvivor',
            creator=PongSurvivor,
            configuration=environment_configurations.classic_one_ball,
    ):
        self.name = name
        self.creator = creator
        self.configuration = configuration


class ReinforcementLearningConfiguration:
    def __init__(
            self,
            architecture_name=None,
            architecture_configuration=None,
            stopping_criterion=None,

            train_batch_size=None,
            mini_batch_size_per_learner=None,
            sgd_minibatch_size=None,
            num_sgd_iter=None,
            learning_rate=None,

            num_learners=None,
            num_cpus_per_learner=None,
            num_gpus_per_learner=None,
            num_env_runners=None,
            evaluation_num_env_runners=None,
            evaluation_interval=None,
    ):
        self.architecture_name = architecture_name
        self.architecture_configuration = architecture_configuration
        self.stopping_criterion = stopping_criterion

        self.train_batch_size = train_batch_size
        self.mini_batch_size_per_learner = mini_batch_size_per_learner
        self.sgd_minibatch_size = sgd_minibatch_size
        self.num_sgd_iter = num_sgd_iter
        self.learning_rate = learning_rate

        self.num_learners = num_learners
        self.num_cpus_per_learner = num_cpus_per_learner
        self.num_gpus_per_learner = num_gpus_per_learner
        self.num_env_runners = num_env_runners
        self.evaluation_num_env_runners = evaluation_num_env_runners
        self.evaluation_interval = evaluation_interval


class GenerationEpisodeVideosConfiguration:
    def __init__(
            self,
            generation_videos_workers_number=1,
            number_videos_episodes_per_worker=10,
    ):
        self.generation_videos_workers_number = generation_videos_workers_number
        self.number_videos_episodes_per_worker = number_videos_episodes_per_worker


class GenerationDatasetsConfiguration:
    def __init__(
            self,
            workers_number=1,
            number_episodes_per_worker=10,
            number_iterations=100,
    ):
        self.workers_number = workers_number
        self.number_episodes_per_worker = number_episodes_per_worker
        self.number_iterations = number_iterations


class LinearPerturbationLatentSpaceConfiguration:
    def __init__(
            self,
            architecture=Dense,
            accelerator=None,
            max_time=None,
            batch_size=None,
            number_worker_datamodule=None,
            check_val_every_n_epoch=None,
    ):
        self.architecture = architecture
        self.accelerator = accelerator
        self.max_time = max_time
        self.batch_size = batch_size
        self.number_worker_datamodule = number_worker_datamodule
        self.check_val_every_n_epoch = check_val_every_n_epoch


class ExperimentConfiguration:
    def __init__(
            self,
            experiment_name,
            ray_local_mode=False,
            environment_configuration: EnvironmentConfiguration = EnvironmentConfiguration(),
            do_reinforcement_learning=True,
            reinforcement_learning_configuration: ReinforcementLearningConfiguration = ReinforcementLearningConfiguration(),
            do_generation_videos=True,
            generation_videos_configuration: GenerationEpisodeVideosConfiguration = GenerationEpisodeVideosConfiguration(),
            do_generation_datasets=False,
            generation_datasets_configuration: GenerationEpisodeVideosConfiguration = GenerationEpisodeVideosConfiguration(),
            do_linear_perturbation=False,
            linear_perturbation_configuration: LinearPerturbationLatentSpaceConfiguration = LinearPerturbationLatentSpaceConfiguration(),
    ):
        self.experiment_name = experiment_name
        self.ray_local_mode = ray_local_mode

        self.do_reinforcement_learning = do_reinforcement_learning
        self.do_generation_videos = do_generation_videos
        self.do_generation_datasets = do_generation_datasets
        self.do_linear_perturbation = do_linear_perturbation

        self.environment: EnvironmentConfiguration = environment_configuration
        self.reinforcement_learning: ReinforcementLearningConfiguration = reinforcement_learning_configuration
        self.generation_episode_videos: GenerationEpisodeVideosConfiguration = generation_videos_configuration
        self.Generation_datasets_configuration: GenerationDatasetsConfiguration = generation_datasets_configuration
        self.linear_perturbation_latent_space: LinearPerturbationLatentSpaceConfiguration = linear_perturbation_configuration
