import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import Timer
from datetime import timedelta
import torch


class Autoencoding(pl.LightningModule):
    def __init__(self, name, architecture, input_shape, output_shape, save_path=None, tensorboard_path=None):
        super(Autoencoding, self).__init__()
        self.name = name
        self.save_path = save_path
        self.tensorboard_path = tensorboard_path

        self.coef_kullback_leibler_loss = 0.1

        self.model = architecture(input_shape, output_shape)
        self.criterion = nn.MSELoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def kullback_leibler_loss(self, distribution, latent_code):
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(latent_code, device=latent_code.device),
            scale_tril=torch.eye(latent_code.shape[-1], device=latent_code.device).unsqueeze(
                0).expand(
                latent_code.shape[0], -1, -1),
        )
        kullback_leibler_loss = torch.distributions.kl.kl_divergence(distribution, std_normal).mean()
        return kullback_leibler_loss

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, distribution, latent_code = self(x)

        reconstruction_loss = self.criterion(x_hat, x)
        kullback_leibler_loss = self.kullback_leibler_loss(distribution=distribution, latent_code=latent_code)

        self.log('train_reconstruction_loss', reconstruction_loss)
        self.log('train_kullback_leibler_loss', kullback_leibler_loss)
        loss = reconstruction_loss + self.coef_kullback_leibler_loss * kullback_leibler_loss
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, distribution, latent_code = self(x)

        reconstruction_loss = self.criterion(x_hat, x)
        kullback_leibler_loss = self.kullback_leibler_loss(distribution=distribution, latent_code=latent_code)

        self.log('validation_reconstruction_loss', reconstruction_loss)
        self.log('validation_kullback_leibler_loss', kullback_leibler_loss)
        loss = reconstruction_loss + self.coef_kullback_leibler_loss * kullback_leibler_loss
        self.log('validation_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_hat, distribution, latent_code = self(x)

        reconstruction_loss = self.criterion(x_hat, x)
        kullback_leibler_loss = self.kullback_leibler_loss(distribution=distribution, latent_code=latent_code)

        self.log('test_train_reconstruction_loss', reconstruction_loss)
        self.log('test_kullback_leibler_loss', kullback_leibler_loss)
        loss = reconstruction_loss + self.coef_kullback_leibler_loss * kullback_leibler_loss
        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer

    def learning(
        self,
        data,
        max_epochs=-1,
        max_model_save=3,
        patience=50,
        accelerator='cpu',
        devices='auto',
        max_time=timedelta(days=7),
        check_val_every_n_epoch=1,
    ):
        timer = Timer(duration=max_time)
        logger = TensorBoardLogger(
            name=self.name,
            prefix='supervised/',
            save_dir=self.tensorboard_path,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='validation_loss',
            dirpath=self.save_path,
            filename=str(self.name) + '/model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=max_model_save,
            mode='min',
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
