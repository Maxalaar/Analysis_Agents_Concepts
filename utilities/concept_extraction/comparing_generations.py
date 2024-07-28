import os

import torch
from utilities.data import DataModule, H5Dataset
from utilities.image import save
from utilities.path import PathManager
import numpy as np


def comparing_generations(
    path_manager: PathManager,
    data_path,
    x_name,
    y_name,
    sample_size,
    model,
    environment,
):
    data = H5Dataset(data_path, x_name, y_name)
    random_indices = np.sort(np.random.choice(len(data), sample_size, replace=True))
    x, y = data[random_indices]

    with torch.no_grad():
        y_hat = model(x.to(model.device))
        y_hat = y_hat.to('cpu')

    y = y.numpy()
    y_hat = y_hat.numpy()

    for i in range(sample_size):
        environment.set_from_observation(y[i])
        representation = environment.render()
        environment.set_from_observation(y_hat[i])
        representation_generation = environment.render()

        comparison = np.concatenate((representation, representation_generation), axis=1)

        save(
            image=comparison,
            directory=os.path.join(path_manager.generations_directory, str(model.name)),
            name=str(i) + '.png',
        )

