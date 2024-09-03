import os

import torch
import torch.distributions as distributions
from utilities.data import DataModule, H5Dataset
from utilities.image import save
from utilities.path import PathManager
import numpy as np


def generation_observations(
    path_manager: PathManager,
    sample_size,
    model,
    environment,
):
    gaussian_distribution = distributions.MultivariateNormal(torch.zeros(model.model.input_dimension), torch.eye(model.model.input_dimension[0]))
    x = gaussian_distribution.sample((sample_size,))

    with torch.no_grad():
        y_hat = model(x.to(model.device))
        if type(y_hat) is not tuple:
            y_hat = y_hat.to('cpu')
        else:
            y_hat = y_hat[0].to('cpu')

    y_hat = y_hat.numpy()

    for i in range(sample_size):
        environment.set_from_observation(y_hat[i])
        representation_generation = environment.render()

        save(
            image=representation_generation,
            directory=os.path.join(path_manager.generations_directory, str(model.name)),
            name=str(i) + '.png',
        )
