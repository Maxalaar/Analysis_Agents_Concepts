from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.typing import TensorType


def kl_divergence(mu, logvar):
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
    return kl_div.mean()


class GaussianNormalizationAutoencoder(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_configuration, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_configuration, name)
        nn.Module.__init__(self)

        self.observation_size = get_preprocessor(observation_space)(observation_space).size
        self.number_outputs = num_outputs
        self.encoding_size = kwargs.get('encoding_size', 32)
        self.activation_function = nn.LeakyReLU()

        self.architecture_encoder = kwargs.get('architecture_encoder', [128, 64]) + [self.encoding_size]
        self.architecture_decoder = kwargs.get('architecture_decoder', [64, 128])
        self.architecture_actor = kwargs.get('architecture_actor', []) + [self.number_outputs]
        self.architecture_critic = kwargs.get('architecture_critic', []) + [1]

        self.encoder_layers = [nn.Linear(self.observation_size, self.architecture_encoder[0]), self.activation_function]
        self.decoder_layers = [nn.Linear(self.encoding_size, self.architecture_decoder[0]), self.activation_function]

        self.actor_layers = [nn.Linear(self.architecture_decoder[-1], self.architecture_actor[0]), self.activation_function]
        self.critic_layers = [nn.Linear(self.architecture_decoder[-1], self.architecture_critic[0]), self.activation_function]

        for i in range(0, len(self.architecture_encoder) - 1):
            self.encoder_layers.append(nn.Linear(self.architecture_encoder[i], self.architecture_encoder[i + 1]))
            self.encoder_layers.append(self.activation_function)

        for i in range(0, len(self.architecture_decoder) - 1):
            self.decoder_layers.append(nn.Linear(self.architecture_decoder[i], self.architecture_decoder[i + 1]))
            self.decoder_layers.append(self.activation_function)

        for i in range(0, len(self.architecture_actor) - 1):
            self.actor_layers.append(nn.Linear(self.architecture_actor[i], self.architecture_actor[i + 1]))
            self.actor_layers.append(self.activation_function)

        for i in range(0, len(self.architecture_critic) - 1):
            self.critic_layers.append(nn.Linear(self.architecture_critic[i], self.architecture_critic[i + 1]))
            self.critic_layers.append(self.activation_function)

        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self.decoder_layers = nn.Sequential(*self.decoder_layers)
        self.actor_layers = nn.Sequential(*self.actor_layers)
        self.critic_layers = nn.Sequential(*self.critic_layers)

        self.gaussian_embedding = None
        self.embedding = None
        self.flatten_observation = None

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self.flatten_observation = input_dict['obs_flat']
        self.gaussian_embedding = self.encoder_layers(self.flatten_observation)
        self.embedding = self.decoder_layers(self.gaussian_embedding)
        action = self.actor_layers(self.embedding)

        return action, []

    def value_function(self):
        value = self.critic_layers(self.embedding)
        return torch.reshape(value, [-1])

    def kullback_leibler_loss(self):
        mu = torch.mean(self.gaussian_embedding, dim=0)
        logvar = torch.log(torch.var(self.gaussian_embedding, dim=0))

        kullback_leibler_loss = kl_divergence(mu, logvar)
        return kullback_leibler_loss

    def custom_loss(self, policy_loss, loss_inputs):
        kullback_leibler_loss = self.kullback_leibler_loss()

        self.metric_kullback_leibler_loss = kullback_leibler_loss.item()
        self.metric_policy_loss = np.mean([loss.item() for loss in policy_loss])

        coef_kullback_leibler_loss = 0.001

        return [policy_loss[0] + coef_kullback_leibler_loss * kullback_leibler_loss]

    def metrics(self) -> Dict[str, TensorType]:
        return dict(kullback_leibler_loss=self.metric_kullback_leibler_loss, policy_loss=self.metric_policy_loss)
