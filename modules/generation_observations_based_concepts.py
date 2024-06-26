import os

import torch
import gymnasium
from ray.rllib.env.env_context import EnvContext

from utilities.path import PathManager
from utilities.lightning.load import load
from utilities.image import save


def generation_observations_based_concepts(path_manager: PathManager, environment, number_elements_per_concept=100):
    model_module = load(
        model_directory=path_manager.lightning_model_directory,
    )
    # number_concepts = model_module.model.input_size
    number_concepts = 10

    for concept_number in range(number_concepts):
        variation_concept = torch.zeros((number_elements_per_concept, number_concepts), dtype=torch.float32).to(model_module.device.type)

        for index, concept_vector in enumerate(variation_concept):
            interval = 1
            concept_vector[concept_number] = index * 2 * (interval / number_elements_per_concept) - interval

        with torch.no_grad():
            observations_generated = model_module(variation_concept)
            observations_generated = observations_generated.to('cpu')
            observations_generated = observations_generated.numpy()

        for observation_number, observation in enumerate(observations_generated):
            environment.set_from_observation(observation)
            representation = environment.render()
            save(
                image=representation,
                directory=os.path.join(path_manager.concepts_directory, 'concept_' + str(concept_number)),
                name=str(observation_number) + '.png',
            )

