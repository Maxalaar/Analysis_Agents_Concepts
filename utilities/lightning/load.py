import os

from utilities.lightning.supervised import Supervised
from utilities.path import find_latest_file


def load(model_directory, learning_type):
    checkpoint_path = os.path.join(model_directory, find_latest_file(model_directory))
    model_module = learning_type.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    return model_module
