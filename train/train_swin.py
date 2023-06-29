import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import lightning.pytorch as pl
from datamodule.chirps_cmip6 import ChirpsDataModule, ChirpsCmip6DataModule
from model.swin2sr import Swin2SRLight


def main():
    # 1. Create a model
    model = Swin2SRLight(upscale=5, img_size=(32, 32), in_chans=1,
                window_size=8, img_range=1., depths=[6, 6, 6, 6],
                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

    # 2. Create a datamodule
    datamodule = ChirpsDataModule(data_dir="/dccstor/weathergenerator/users/dilermando/downscaling/datamodule/dataset/high-low",
                                   batch_size=128,
                                   transforms=True)

    # 3. Create a trainer
    trainer = pl.Trainer(
        max_epochs=100,
        logger=pl.loggers.TensorBoardLogger('logs/', name='swin')
    )

    # 4. Train the model
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()