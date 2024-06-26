import pytorch_lightning as pl
import torch
import torch.nn as nn


class Simple(pl.LightningModule):
    def __init__(self, input_dimension, output_dimension):
        super(Simple, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        input_size = torch.prod(torch.tensor(self.input_dimension)).item()
        output_size = torch.prod(torch.tensor(self.output_dimension)).item()

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        x = torch.reshape(x, (-1,) + self.output_dimension)
        return x
