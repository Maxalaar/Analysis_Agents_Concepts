import os

from utilities.lightning.module import Supervised
from utilities.path import find_latest_file


def load(model_directory):
    checkpoint_path = os.path.join(model_directory, find_latest_file(model_directory))
    model_module = Supervised.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    return model_module
