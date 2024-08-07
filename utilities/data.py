import os
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, StackDataset
import pytorch_lightning as pl
from torch.utils.data import Dataset


def save_data_to_h5(filename, dataset_name, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, 'a') as h5file:
        if dataset_name in h5file:
            # Dataset exists, append data
            dataset = h5file[dataset_name]
            dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
            dataset[-data.shape[0]:] = data
        else:
            # Dataset does not exist, create it
            max_shape = (None,) + data.shape[1:]  # None allows unlimited growth in the first dimension
            dataset = h5file.create_dataset(dataset_name, data=data, maxshape=max_shape)


def get_h5_shapes(path, name):
    shape = None
    with h5py.File(path, 'r') as file:
        if name in file:
            dataset = file[name]
            shape = dataset.shape
        else:
            print('Dataset ' + str(name) + ' not found in the file.')
    return shape


def split_dataset(dataset, split_ratio=0.2):
    split_len = int(len(dataset) * split_ratio)
    base_len = len(dataset) - split_len
    train_data, test_data = random_split(dataset, [base_len, split_len])
    return train_data, test_data


class H5Dataset(Dataset):
    def __init__(self, path, input_key, output_key=None):
        self.path = path
        self.input_key = input_key
        self.output_key = output_key
        self.file = h5py.File(self.path, 'r')
        self.input_data = self.file[self.input_key][:]
        self.output_data = self.file[self.output_key][:] if self.output_key else None

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input = torch.tensor(self.input_data[idx], dtype=torch.float32)
        if self.output_data is None:
            return input
        else:
            output = torch.tensor(self.output_data[idx], dtype=torch.float32)
            return input, output

    def __del__(self):
        self.file.close()


class DataModule(pl.LightningDataModule):
    def __init__(self, path, x_name, y_name=None, batch_size=32, test_ratio=0.2, number_workers=1):
        super().__init__()
        self.path = path
        self.x_name = x_name
        self.y_name = y_name
        self.number_workers = number_workers

        self.x_shape = get_h5_shapes(self.path, self.x_name)[1:]
        if self.y_name is None:
            self.y_shape = None
        else:
            self.y_shape = get_h5_shapes(self.path, self.y_name)[1:]

        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.train_data = None
        self.test_data = None

    def setup(self, stage=None):
        dataset = H5Dataset(
            path=self.path,
            input_key=self.x_name,
            output_key=self.y_name,
        )

        self.train_data, self.test_data = split_dataset(dataset, self.test_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.number_workers)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.number_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.number_workers)
