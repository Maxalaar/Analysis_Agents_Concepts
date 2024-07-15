import h5py
import dask.array as da

from utilities.custom_scaling import custom_scaling
from utilities.path import PathManager


def singular_value_decomposition_embeddings(path_manager: PathManager,  number_basis_vectors=None):
    with h5py.File(path_manager.embeddings_dataset, 'a') as file:
        if 'embeddings_singular_basis' in file:
            del file['embeddings_singular_basis']

        embeddings = file['embeddings']
        embeddings = da.from_array(embeddings)

        u, s, vt = da.linalg.svd(embeddings)

        if number_basis_vectors is None:
            v_reduce = vt.T
        else:
            v_reduce = vt.T[:, :number_basis_vectors]

        embeddings_singular_basis = da.dot(embeddings, v_reduce)

        column_means = embeddings_singular_basis.mean(axis=0)
        embeddings_singular_basis = embeddings_singular_basis - column_means

        for column_index in range(embeddings_singular_basis.shape[1]):
            embeddings_singular_basis[:, column_index] = custom_scaling(embeddings_singular_basis[:, column_index])

        embeddings_singular_basis = embeddings_singular_basis.compute()

        file['embeddings_singular_basis'] = embeddings_singular_basis
