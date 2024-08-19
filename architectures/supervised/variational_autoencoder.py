from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.typing import TensorType


class VariationalAutoencoder(pl.LightningModule):
    def __init__(self, input_dimension, output_dimension, **kwargs):
        super(VariationalAutoencoder, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_size = torch.prod(torch.tensor(self.input_dimension)).item()
        self.output_size = torch.prod(torch.tensor(self.output_dimension)).item()
        self.activation_function = nn.LeakyReLU()

        self.architecture_encoder = kwargs.get('architecture_encoder', [246, 128])
        self.encoding_size = kwargs.get('encoding_size', 64)
        self.architecture_decoder = kwargs.get('architecture_decoder', [128, 246]) + [self.output_size]

        self.encoder_layers = [nn.Linear(self.input_size, self.architecture_encoder[0]), self.activation_function]
        self.decoder_layers = [nn.Linear(self.encoding_size, self.architecture_decoder[0]), self.activation_function]

        for i in range(0, len(self.architecture_encoder) - 1):
            self.encoder_layers.append(nn.Linear(self.architecture_encoder[i], self.architecture_encoder[i + 1]))
            self.encoder_layers.append(self.activation_function)

        for i in range(0, len(self.architecture_decoder) - 1):
            self.decoder_layers.append(nn.Linear(self.architecture_decoder[i], self.architecture_decoder[i + 1]))
            self.decoder_layers.append(self.activation_function)

        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self.decoder_layers = nn.Sequential(*self.decoder_layers)

        self.mean_layer = nn.Linear(self.architecture_encoder[-1], self.encoding_size)
        self.log_variance_layer = nn.Linear(self.architecture_encoder[-1], self.encoding_size)

        self.softplus = nn.Softplus()

    def encode(self, flatten_observation, epsilon: float = 1e-8):
        x = self.encoder_layers(flatten_observation)

        mean = self.mean_layer(x)
        log_variance = self.log_variance_layer(x)
        scale = self.softplus(log_variance) + epsilon
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)

    def reparameterize(self, distributions):
        return distributions.rsample()

    def decode(self, latent_code):
        x = self.decoder_layers(latent_code)
        return x

    def forward(self, x) -> (TensorType, List[TensorType]):
        distribution = self.encode(x)
        latent_code = self.reparameterize(distribution)
        x = self.decode(latent_code)
        return x, distribution, latent_code
