from ray.tune.registry import register_env

from environments.pong_survivor.pong_survivor import PongSurvivor


def register_environments():
    register_env(name='PongSurvivor', env_creator=PongSurvivor)
