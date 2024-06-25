import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utilities.data import DataModule
from utilities.lightning_module import Supervised


def train(
        experiment_name,
        data_path,
        x_name,
        y_name,
        architecture,
        save_path,
        max_epochs=-1,
        max_model_save=3,
        save_interval=5,
        patience=50,
        accelerator='cpu',
        devices='auto',
):
    data_module = DataModule(data_path, x_name, y_name)
    model = architecture(data_module.x_shape, data_module.y_shape)
    model_module = Supervised(model)

    logger = TensorBoardLogger(
        save_dir=save_path + '/tensorboard/',
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath=save_path + '/model/',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=max_model_save,
        mode='min',
        save_last=True,
        every_n_epochs=save_interval,
    )
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=patience,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=accelerator,
        devices=devices,
    )

    trainer.fit(model_module, data_module)
