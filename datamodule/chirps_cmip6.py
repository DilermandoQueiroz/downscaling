import lightning.pytorch as pl
import torch
import numpy as np
from torchvision import transforms

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

        self.chirps[np.isnan(self.chirps)] = 0
        self.cmip6[np.isnan(self.cmip6)] = 0

        # if chirps and cmip6 are all zero, remove it
        self.chirps2 = []
        self.cmip62 = []
        for i in range(len(self.chirps)):
            if self.chirps[i].sum() != 0 and self.cmip6[i].sum() != 0:
                self.chirps2.append(self.chirps[i])
                self.cmip62.append(self.cmip6[i])
        
        self.chirps = np.array(self.chirps2)
        self.cmip6 = np.array(self.cmip62)
        
        # find mean and std
        self.chirps_mean = self.chirps.mean()
        self.chirps_std = self.chirps.std()
        self.cmip6_mean = self.cmip6.mean()
        self.cmip6_std = self.cmip6.std()

        self.transform_chirps = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.chirps_mean], std=[self.chirps_std])
        ])

        self.transform_cmip6 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.cmip6_mean], std=[self.cmip6_std])
        ])


    def __len__(self):
        """Length of the dataset.
        """
        return len(self.chirps)

    def __getitem__(self, index):
        """Get item.
        """
        chirps = self.chirps[index]
        chirps = self.transform_chirps(chirps)
        cmip6 = self.cmip6[index]
        cmip6 = self.transform_cmip6(cmip6)

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


