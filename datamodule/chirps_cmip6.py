import lightning.pytorch as pl
import torch
import numpy as np

class ChirpsCmip6(torch.utils.data.Dataset):
    """ChirpsCmip6 dataset.
    """
    def __init__(self, data_dir="dataset/high-low", type='train') -> None:
        super().__init__()
        self.data_dir = data_dir
        self.type = type

        if type == 'train':
            self.cmip6 = np.load(f'{data_dir}/train_cmip6.npy')
            self.chirps = np.load(f'{data_dir}/train_chirps.npy')
        elif type == 'val':
            self.cmip6 = np.load(f'{data_dir}/val_cmip6.npy')
            self.chirps = np.load(f'{data_dir}/val_chirps.npy')
        elif type == 'test':
            self.cmip6 = np.load(f'{data_dir}/test_cmip6.npy')
            self.chirps = np.load(f'{data_dir}/test_chirps.npy')
        else:
            raise ValueError(f'Invalid type: {type} (must be train, val or test)')

        self.chirps = [x for x in self.chirps if np.isnan(x).sum() == 0]
        self.cmip6 = [x for x in self.cmip6 if np.isnan(x).sum() == 0]
        self.chirps = np.array(self.chirps)
        self.cmip6 = np.array(self.cmip6)

    def __len__(self):
        """Length of the dataset.
        """
        return len(self.chirps)

    def __getitem__(self, index):
        """Get item.
        """
        chirps = self.chirps[index]
        cmip6 = self.cmip6[index]

        return chirps, cmip6


class ChirpsCmip6DataModule(pl.LightningDataModule):

    def __init__(self, data_dir="dataset/high-low", batch_size=32) -> None:

        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        """Prepare data.
        """
        ...

    def setup(self, stage=None):
        """Setup data.

        Args:
            stage (str, optional): Stage. Defaults to None.
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ChirpsCmip6(data_dir=self.data_dir, type='train')
            self.val_dataset = ChirpsCmip6(data_dir=self.data_dir, type='val')
        if stage == 'test' or stage is None:
            self.test_dataset = ChirpsCmip6(data_dir=self.data_dir, type='test')


    def train_dataloader(self):
        """Training dataloader.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        """Validation dataloader.
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        """Test dataloader.
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


