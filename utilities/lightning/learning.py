from datetime import timedelta

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import Timer

from utilities.data import DataModule
from utilities.lightning.supervised import Supervised


def train(
        data_path,
        tensorboard_path,
        model_path,
        x_name,
        y_name,
        architecture,
        model_name,
        max_epochs=-1,
        max_model_save=3,
        save_interval=5,
        patience=50,
        accelerator='cpu',
        devices='auto',
        batch_size=32,
        max_time=timedelta(days=7),
        number_worker_datamodule=1,
        check_val_every_n_epoch=1,
):
    data_module = DataModule(data_path, x_name, y_name, batch_size=batch_size, number_workers=number_worker_datamodule)
    model = architecture(data_module.x_shape, data_module.y_shape)
    model_module = Supervised(model)

    timer = Timer(duration=max_time)
    logger = TensorBoardLogger(
        name=model_name,
        prefix='lightning/',
        save_dir=tensorboard_path,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath=model_path,
        filename=str(model_name) + '/model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=max_model_save,
        mode='min',
        every_n_epochs=save_interval,
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

    trainer.fit(model=model_module, datamodule=data_module)
