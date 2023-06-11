import lightning.pytorch as pl
from datamodule.chirps_cmip6 import ChirpsDataModule
from model.unet import UnetModule


def main():
    # 1. Create a model
    model = UnetModule()

    # 2. Create a datamodule
    datamodule = ChirpsDataModule()

    # 3. Create a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=50,
        progress_bar_refresh_rate=20,
        log_every_n_steps=1,
        logger=pl.loggers.TensorBoardLogger('logs/', name='unet')
    )

    # 4. Train the model
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()