import torch

from utilities.path import PathManager
from utilities.lightning.load import load


def generation_observations_based_concepts(path_manager: PathManager):
    model_module = load(
        model_directory=path_manager.lightning_model_directory,
    )


    x = model_module(torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32).to(model_module.device.type))
