import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import Timer
from datetime import timedelta


class Supervised(pl.LightningModule):
    def __init__(self, name, architecture, input_shape, output_shape, save_path=None, tensorboard_path=None):
        super(Supervised, self).__init__()
        self.name = name
        self.save_path = save_path
        self.tensorboard_path = tensorboard_path

        self.model = architecture(input_shape, output_shape)
        self.criterion = nn.MSELoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('validation_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def learning(
        self,
        data,
        max_epochs=-1,
        max_model_save=3,
        save_time_interval=timedelta(minutes=1),
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
            # train_time_interval=save_time_interval,
            # every_n_train_steps=1000,
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
            # val_check_interval=1000,
            # log_every_n_steps=1000,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback, timer],
            accelerator=accelerator,
            devices=devices,
        )

        trainer.fit(model=self, datamodule=data)
