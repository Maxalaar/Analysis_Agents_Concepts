import torch
import torch.nn as nn
import pytorch_lightning as pl


class Dense(pl.LightningModule):
    def __init__(self, input_dimension, output_dimension, layer_configuration=None, activation_fn=nn.LeakyReLU):
        super(Dense, self).__init__()
        if layer_configuration is None:
            layer_configuration = [64, 64, 64]
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_size = torch.prod(torch.tensor(self.input_dimension)).item()
        self.output_size = torch.prod(torch.tensor(self.output_dimension)).item()

        self.flatten = nn.Flatten()

        # Create layers dynamically based on architecture
        layers = []
        in_features = self.input_size
        for out_features in layer_configuration:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation_fn())
            in_features = out_features

        # Add the final layer
        layers.append(nn.Linear(in_features, self.output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        x = torch.reshape(x, (-1,) + self.output_dimension)
        return x
