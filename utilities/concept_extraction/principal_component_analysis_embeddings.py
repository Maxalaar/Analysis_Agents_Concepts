import h5py
import dask.array as da
from dask_ml.decomposition import PCA
from dask_ml.preprocessing import StandardScaler

from utilities.concept_extraction.custom_scaling import custom_scaling
from utilities.path import PathManager


def principal_component_analysis_embeddings(path_manager: PathManager,  number_basis_vectors=None):
    with h5py.File(path_manager.embeddings_dataset, 'a') as file:
        if 'embeddings_principal_component_bases' in file:
            del file['embeddings_principal_component_bases']

        embeddings = file['embeddings']
        embeddings = da.from_array(embeddings)

        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        pca = PCA(n_components=number_basis_vectors)
        embeddings_principal_component_bases = pca.fit_transform(embeddings)

        column_means = embeddings_principal_component_bases.mean(axis=0)
        embeddings_principal_component_bases = embeddings_principal_component_bases - column_means

        for column_index in range(embeddings_principal_component_bases.shape[1]):
            embeddings_principal_component_bases[:, column_index] = custom_scaling(embeddings_principal_component_bases[:, column_index])

        embeddings_principal_component_bases = embeddings_principal_component_bases.compute()

        file['embeddings_principal_component_bases'] = embeddings_principal_component_bases
