from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import Timer
from datetime import timedelta
from geomloss import SamplesLoss


class Perturbation(pl.LightningModule):
    def __init__(
        self,
        name,
        architecture,
        observation_shape,
        embedding_shape,
        embeddings_to_observations,
        perturbations_number,
        perturbation_magnitude,
        save_path=None,
        tensorboard_path=None,
    ):
        super(Perturbation, self).__init__()
        self.name = name
        self.save_path = save_path
        self.tensorboard_path = tensorboard_path

        self.perturbations_number = perturbations_number
        self.model = architecture(torch.prod(torch.tensor(observation_shape)).item() * 2, (self.perturbations_number+1,))
        self.perturbations_matrix = nn.Parameter(torch.randn(perturbations_number, *embedding_shape))

        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_regression = nn.MSELoss()

        self.embeddings_to_observations = embeddings_to_observations
        self.perturbation_magnitude = perturbation_magnitude
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        perturbation_class_number = (torch.randint(0, self.perturbations_number, (x.shape[0],))).to(self.device)
        random_magnitude = ((torch.rand((x.shape[0], 1)) * 2 - 1) * self.perturbation_magnitude).to(self.device)
        perturbation_class_one_hot = torch.nn.functional.one_hot(perturbation_class_number, num_classes=self.perturbations_number).to(torch.float32)

        x_perturb = x + self.perturbations_matrix[perturbation_class_number] * random_magnitude
        return self.model(torch.cat((self.embeddings_to_observations(x), self.embeddings_to_observations(x_perturb)), dim=1)), perturbation_class_one_hot, random_magnitude

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat, perturbation_class_one_hot, random_magnitude = self.forward(x)
        loss_classification = self.criterion_classification(y_hat[:, :self.perturbations_number], perturbation_class_one_hot)
        loss_regression = self.criterion_regression(y_hat[:, self.perturbations_number], random_magnitude.squeeze(1))
        loss = loss_classification + 0.25 * loss_regression

        self.log('train_loss_classification', loss_classification)
        self.log('train_loss_regression', loss_regression)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat, perturbation_class_one_hot, random_magnitude = self.forward(x)
        loss_classification = self.criterion_classification(y_hat[:, :self.perturbations_number],
                                                            perturbation_class_one_hot)
        loss_regression = self.criterion_regression(y_hat[:, self.perturbations_number], random_magnitude.squeeze(1))
        loss = loss_classification + 0.25 * loss_regression

        self.log('validation_loss_classification', loss_classification)
        self.log('validation_loss_regression', loss_regression)
        self.log('validation_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat, perturbation_class_one_hot, random_magnitude = self.forward(x)
        loss_classification = self.criterion_classification(y_hat[:, :self.perturbations_number],
                                                            perturbation_class_one_hot)
        loss_regression = self.criterion_regression(y_hat[:, self.perturbations_number], random_magnitude.squeeze(1))
        loss = loss_classification + 0.25 * loss_regression

        self.log('test_loss_classification', loss_classification)
        self.log('test_loss_regression', loss_regression)
        self.log('test_loss', loss)
        return loss

    def on_after_backward(self):
        with torch.no_grad():
            self.perturbations_matrix.div_(torch.norm(self.perturbations_matrix, dim=1, keepdim=True))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=[
                {'params': self.model.parameters()},
                {'params': self.perturbations_matrix}
            ],
            lr=1e-3,
        )
        return optimizer

    def learning(
        self,
        data,
        max_epochs=-1,
        max_model_save=3,
        save_time_interval=timedelta(minutes=30),
        patience=50,
        accelerator='cpu',
        devices='auto',
        max_time=timedelta(days=7),
        check_val_every_n_epoch=1,
    ):
        timer = Timer(duration=max_time)
        logger = TensorBoardLogger(
            name=self.name,
            prefix='perturbation/',
            save_dir=self.tensorboard_path,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='validation_loss',
            dirpath=self.save_path,
            filename=str(self.name) + '/model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=max_model_save,
            mode='min',
            # train_time_interval=save_time_interval,
            every_n_epochs=1,
            save_last=False,
        )
        early_stop_callback = EarlyStopping(
            monitor='validation_loss',
            patience=patience,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback, timer],
            accelerator=accelerator,
            devices=devices,
        )

        trainer.fit(model=self, datamodule=data)
