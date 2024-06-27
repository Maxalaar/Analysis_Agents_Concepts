from ray.rllib.models import ModelCatalog

from architectures.reinforcement.dense import Dense


def register_architectures():
    ModelCatalog.register_custom_model('dense', Dense)
