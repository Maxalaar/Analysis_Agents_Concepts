from ray.rllib.models import ModelCatalog

from architectures.reinforcement.dense import Dense
from architectures.reinforcement.gaussian_normalization_autoencoder import GaussianNormalizationAutoencoder
from architectures.reinforcement.variational_autoencoder import VariationalAutoencoder


def register_architectures():
    ModelCatalog.register_custom_model('dense', Dense)
    ModelCatalog.register_custom_model('variational_autoencoder', VariationalAutoencoder)
    ModelCatalog.register_custom_model('gaussian_normalization_autoencoder', GaussianNormalizationAutoencoder)
