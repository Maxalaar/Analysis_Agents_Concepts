from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.typing import TensorType


# class VariationalAutoencoder(TorchModelV2, nn.Module):
#     def __init__(self, observation_space, action_space, num_outputs, model_configuration, name, **kwargs):
#         TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_configuration, name)
#         nn.Module.__init__(self)
#
#         self.observation_size = get_preprocessor(observation_space)(observation_space).size
#         self.action_size = num_outputs  # get_preprocessor(action_space)(action_space).size
#         self.encoding_size = kwargs.get('encoding_size', 32)
#         self.activation_function = nn.LeakyReLU()  # nn.ReLU()
#
#         self.architecture_encoder = kwargs.get('architecture_encoder', [128, 64])
#         self.architecture_decoder = kwargs.get('architecture_decoder', [64, 128]) + [self.observation_size]
#         self.architecture_actor = kwargs.get('architecture_actor', []) + [self.action_size]
#         self.architecture_critic = kwargs.get('architecture_critic', []) + [1]
#
#         self.encoder_layers = [nn.Linear(self.observation_size, self.architecture_encoder[0]), self.activation_function]
#         self.decoder_layers = [nn.Linear(self.encoding_size, self.architecture_decoder[0]), self.activation_function]
#
#         self.actor_layers = [nn.Linear(self.architecture_decoder[-1], self.architecture_actor[0]), self.activation_function]
#         self.critic_layers = [nn.Linear(self.architecture_decoder[-1], self.architecture_critic[0]), self.activation_function]
#
#         for i in range(0, len(self.architecture_encoder) - 1):
#             self.encoder_layers.append(nn.Linear(self.architecture_encoder[i], self.architecture_encoder[i + 1]))
#             self.encoder_layers.append(self.activation_function)
#
#         for i in range(0, len(self.architecture_decoder) - 1):
#             self.decoder_layers.append(nn.Linear(self.architecture_decoder[i], self.architecture_decoder[i + 1]))
#             self.decoder_layers.append(self.activation_function)
#
#         for i in range(0, len(self.architecture_actor) - 1):
#             self.actor_layers.append(nn.Linear(self.architecture_actor[i], self.architecture_actor[i + 1]))
#             self.actor_layers.append(self.activation_function)
#
#         for i in range(0, len(self.architecture_critic) - 1):
#             self.critic_layers.append(nn.Linear(self.architecture_critic[i], self.architecture_critic[i + 1]))
#             self.critic_layers.append(self.activation_function)
#
#         self.encoder_layers = nn.Sequential(*self.encoder_layers)
#         self.decoder_layers = nn.Sequential(*self.decoder_layers)
#         self.actor_layers = nn.Sequential(*self.actor_layers)
#         self.critic_layers = nn.Sequential(*self.critic_layers)
#
#         self.mean_layer = nn.Linear(self.architecture_encoder[-1], self.encoding_size)
#         self.log_variance_layer = nn.Linear(self.architecture_encoder[-1], self.encoding_size)
#
#         self.softplus = nn.Softplus()
#
#         self.flatten_observation = None
#         self.latent_code = None
#         self.encode_mean = None
#         self.encode_log_variance = None
#         self.metric_kullback_leibler_loss = None
#         self.metric_policy_loss = None
#         self.current_distribution = None
#         self.embedding = None
#
#     def encode(self, flatten_observation, epsilon: float = 1e-8):
#         x = self.encoder_layers(flatten_observation)
#
#         mean = self.mean_layer(x)
#         log_variance = self.log_variance_layer(x)
#         scale = self.softplus(log_variance) + epsilon
#         scale_tril = torch.diag_embed(scale)
#
#         return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
#
#     def reparameterize(self, distributions):
#         return distributions.rsample()
#
#     def forward(
#             self,
#             input_dict: Dict[str, TensorType],
#             state: List[TensorType],
#             seq_lens: TensorType,
#     ) -> (TensorType, List[TensorType]):
#         self.flatten_observation = input_dict['obs_flat']
#
#         self.current_distribution = self.encode(self.flatten_observation)
#         self.latent_code = self.reparameterize(self.current_distribution)
#         self.embedding = self.decoder_layers(self.latent_code)
#
#         action = self.actor_layers(self.embedding)
#         return action, []
#
#     def value_function(self):
#         value = self.critic_layers(self.embedding)
#         return torch.reshape(value, [-1])
#
#     def kullback_leibler_loss(self):
#         # gaussian_distribution = torch.distributions.MultivariateNormal(
#         #     torch.zeros_like(self.latent_code, device=self.latent_code.device),
#         #     scale_tril=torch.eye(self.latent_code.shape[-1], device=self.latent_code.device).unsqueeze(0).expand(self.latent_code.shape[0], -1, -1),
#         # )
#         # kullback_leibler_loss = torch.distributions.kl.kl_divergence(self.current_distribution, gaussian_distribution).mean()
#         # return kullback_leibler_loss
#
#         gaussian_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.latent_code[:].shape[1]), torch.eye(self.latent_code[:].shape[1]))
#         kullback_leibler_loss = torch.distributions.kl_divergence(self.current_distribution, gaussian_distribution).mean()
#         return kullback_leibler_loss
#
#     def custom_loss(self, policy_loss, loss_inputs):
#         kullback_leibler_loss = self.kullback_leibler_loss()
#
#         self.metric_kullback_leibler_loss = kullback_leibler_loss.item()
#         self.metric_policy_loss = np.mean([loss.item() for loss in policy_loss])
#
#         return [policy_loss[0] + 0.0001 * kullback_leibler_loss]
#
#     def metrics(self) -> Dict[str, TensorType]:
#         return {
#             'kullback_leibler_loss': self.metric_kullback_leibler_loss,
#             'policy_loss': self.metric_policy_loss,
#         }
#
#     def get_latent_space(self, observations):
#         with torch.no_grad():
#             input_dict = {'obs': observations, 'obs_flat': observations}
#             self.forward(input_dict, None, None)
#             return self.latent_code


class VariationalAutoencoder(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_configuration, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_configuration, name)
        nn.Module.__init__(self)

        self.observation_size = get_preprocessor(observation_space)(observation_space).size
        self.action_size = num_outputs  # get_preprocessor(action_space)(action_space).size
        self.encoding_size = kwargs.get('encoding_size', 32)
        self.activation_function = nn.LeakyReLU()  # nn.ReLU()

        self.architecture_encoder = kwargs.get('architecture_encoder', [128, 64])
        self.architecture_decoder = kwargs.get('architecture_decoder', [64, 128]) + [self.observation_size]
        self.architecture_actor = kwargs.get('architecture_actor', []) + [self.action_size]
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

        self.mean_layer = nn.Linear(self.architecture_encoder[-1], self.encoding_size)
        self.log_variance_layer = nn.Linear(self.architecture_encoder[-1], self.encoding_size)

    def encode(self, flatten_observation, epsilon: float = 1e-8):
        x = self.encoder_layers(flatten_observation)

        mean = self.mean_layer(x)
        log_variance = self.log_variance_layer(x)
        return mean, log_variance

    @staticmethod
    def reparameterize(mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(standard_deviation)
        return eps * standard_deviation + mean

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self.flatten_observation = input_dict['obs_flat']

        self.mean, self.log_variance = self.encode(self.flatten_observation)
        self.latent_code = self.reparameterize(self.mean, self.log_variance)
        self.embedding = self.decoder_layers(self.latent_code)

        action = self.actor_layers(self.embedding)
        return action, []

    def value_function(self):
        value = self.critic_layers(self.embedding)
        return torch.reshape(value, [-1])

    def kullback_leibler_loss(self):
        kullback_leibler_loss = torch.mean(-0.5 * torch.sum(1 + self.log_variance - self.mean ** 2 - self.log_variance.exp(), dim=1), dim=0)
        return kullback_leibler_loss

    def custom_loss(self, policy_loss, loss_inputs):
        kullback_leibler_loss = self.kullback_leibler_loss()

        self.metric_kullback_leibler_loss = kullback_leibler_loss.item()
        self.metric_policy_loss = np.mean([loss.item() for loss in policy_loss])

        return [1 * policy_loss[0] + 0.00001 * kullback_leibler_loss]

    def metrics(self) -> Dict[str, TensorType]:
        return {
            'kullback_leibler_loss': self.metric_kullback_leibler_loss,
            'policy_loss': self.metric_policy_loss,
        }

    def get_latent_space(self, observations):
        with torch.no_grad():
            input_dict = {'obs': observations, 'obs_flat': observations}
            self.forward(input_dict, None, None)
            return self.latent_code
