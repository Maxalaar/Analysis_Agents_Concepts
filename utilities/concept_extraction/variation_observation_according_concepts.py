import os

import numpy as np
import torch

from utilities.data import H5Dataset
from utilities.lightning.perturbation import Perturbation
from utilities.lightning.supervised import Supervised
from utilities.path import PathManager
from utilities.lightning.load import load
from utilities.image import save


def variation_observation_according_concepts(
    path_manager: PathManager,
    data_path,
    x_name,
    perturbator_name,
    generator_name,
    environment,
    number_elements_per_observation=10,
    number_observations_per_concept=10,
):
    data = H5Dataset(data_path, x_name)
    perturbator: Perturbation = load(
        model_directory=path_manager.lightning_models_directory + '/' + str(perturbator_name),
        learning_type=Perturbation,
    )
    generator: Supervised = load(
        model_directory=path_manager.lightning_models_directory + '/' + str(generator_name),
        learning_type=Supervised,
    )
    number_concepts = perturbator.perturbations_number
    perturbation_magnitude = perturbator.perturbation_magnitude

    for concept_number in range(number_concepts):
        concept_vector = perturbator.perturbations_matrix[concept_number].to('cpu')
        random_indices = np.sort(np.random.choice(len(data), number_observations_per_concept, replace=True))
        embeddings = data[random_indices].to('cpu')

        for index_embedding, embedding in enumerate(embeddings):
            for index_variation in range(number_elements_per_observation + 1):
                value_perturbation = (index_variation/float(number_elements_per_observation) * perturbation_magnitude * 2) - perturbation_magnitude
                print(value_perturbation)
                embedding = embedding + (value_perturbation * concept_vector)
                with torch.no_grad():
                    observation = generator(embedding.unsqueeze(0).to(generator.device)).to('cpu').numpy()
                environment.set_from_observation(observation[0])
                representation = environment.render()

                save(
                    image=representation,
                    directory=os.path.join(path_manager.concepts_directory, str(generator.name), 'concept_' + str(concept_number)),
                    name=str(index_embedding) + '_' + str(index_variation) + '.png',
                )
