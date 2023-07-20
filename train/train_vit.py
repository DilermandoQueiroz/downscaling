import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import lightning.pytorch as pl
from datamodule.chirps_cmip6 import ChirpsDataModule, ChirpsCmip6DataModule
from model.unet import UnetModule
from model.vit import VisionTransfomerModule


def main():
    # 1. Create a model
    model = VisionTransfomerModule(img_size=(32,32), in_channels=1, out_channels=1, history=1)

    # 2. Create a datamodule
    datamodule = ChirpsCmip6DataModule(data_dir='dir',
                                       transforms=True)

    # 3. Create a trainer
    trainer = pl.Trainer(
        max_epochs=100,
        logger=pl.loggers.TensorBoardLogger('logs/', name='vit-bias')
    )

    # 4. Train the model
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()