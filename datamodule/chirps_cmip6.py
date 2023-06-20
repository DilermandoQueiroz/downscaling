import lightning.pytorch as pl
import torch
import numpy as np
from torchvision import transforms
from copy import deepcopy
import random

class BlurChirps(torch.utils.data.Dataset):
    """ BlurChirps dataset.
    """

    def __init__(self, data_dir="dataset/high-low", type='train', transformations=False, scale=5, crop=160) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.type = type

        if type == 'train':
            self.chirps = np.load(f'{data_dir}/train_chirps.npy')
        elif type == 'val':
            self.chirps = np.load(f'{data_dir}/val_chirps.npy')
        elif type == 'test':
            self.chirps = np.load(f'{data_dir}/test_chirps.npy')
        else:
            raise ValueError(f'Invalid type: {type} (must be train, val or test)')
        
        # Transform nan to zero
        self.chirps[np.isnan(self.chirps)] = 0
        
        # Remove images that are all zero
        self.chirps = np.array([image for image in self.chirps if image.sum() != 0])

        # find mean and std
        self.mean = self.chirps.mean()
        self.std = self.chirps.std()
        self.min = self.chirps.min()
        self.max = self.chirps.max()

        self.chirps = (self.chirps - self.min) / (self.max - self.min)

    def __len__(self):
        """Length of the dataset.
        """
        return len(self.chirps)
    
    def __getitem__(self, index):
        """Get item.
        """
        chirps = self.chirps[index]
        chirps = torch.tensor(chirps).unsqueeze(0)
        chirps = transforms.Resize((224, 224), antialias=True)(chirps)
        
        if random.random() < 0.5:
            transform = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            return transform(chirps), torch.tensor(0, dtype=torch.long)
        
        return chirps, torch.tensor(1, dtype=torch.long)
    
class Chirps(torch.utils.data.Dataset):
    """Chirps dataset.
    """

    def __init__(self, data_dir="dataset/high-low", type='train', transformations=False, scale=5, crop=160) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.type = type
        self.crop = crop
        self.scale = scale
        self.transformations = transformations

        if crop % scale != 0:
            raise ValueError(f'the scale need to be a factor of crop')

        if type == 'train':
            self.chirps = np.load(f'{data_dir}/train_chirps.npy')
        elif type == 'val':
            self.chirps = np.load(f'{data_dir}/val_chirps.npy')
        elif type == 'test':
            self.chirps = np.load(f'{data_dir}/test_chirps.npy')
        else:
            raise ValueError(f'Invalid type: {type} (must be train, val or test)')
        

        self.image_size = self.chirps.shape[1]

        if crop > self.image_size:
            raise f"The crop size {crop} cannot be largen the image size {self.image_size}"

        # Transform nan to zero
        self.chirps[np.isnan(self.chirps)] = 0
        
        # Remove images that are all zero
        self.chirps = np.array([image for image in self.chirps if image.sum() != 0])

        # find mean and std
        self.mean = self.chirps.mean()
        self.std = self.chirps.std()
        self.min = self.chirps.min()
        self.max = self.chirps.max()

        self.chirps = (self.chirps - self.min) / (self.max - self.min)
        
        if crop < self.image_size:
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
            
        elif self.transformations and crop == self.image_size:
            self.transform = transforms.Compose([
                transforms.Pad(8),
                transforms.RandomCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])

    def __len__(self):
        """Length of the dataset.
        """
        return len(self.chirps)
    
    def __getitem__(self, index):
        """Get item.
        """
        chirps = self.chirps[index]
        chirps = torch.tensor(chirps).unsqueeze(0)
        
        new_size = int(self.image_size / self.scale)
        
        chirps_low = transforms.Resize((new_size, new_size), antialias=True)(chirps)
        chirps_low = transforms.Resize((self.image_size, self.image_size), antialias=True)(chirps_low)

        if self.transformations:
            images = torch.cat((chirps_low.unsqueeze(0), chirps.unsqueeze(0)), 0)
            transformed_images = self.transform(images)
            chirps_low = transformed_images[0]
            chirps = transformed_images[1]

        return chirps_low.to(torch.float32), chirps.to(torch.float32)

class ChirpsCmip6(torch.utils.data.Dataset):
    """ChirpsCmip6 dataset.
    """
    def __init__(self, data_dir="dataset/high-low", type='train', transformations=False) -> None:
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
        self.chirps_max = self.chirps.max()
        self.chirps_min = self.chirps.min()
        self.cmip6_max = self.cmip6.max()
        self.cmip6_min = self.cmip6.min()

        self.chirps = (self.chirps - self.chirps_min) / (self.chirps_max - self.chirps_min)
        self.cmip6 = (self.cmip6 - self.cmip6_min) / (self.cmip6_max - self.cmip6_min)
        
        if transformations:
            self.transforms = transforms.Compose([
                transforms.Pad(8),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            self.transforms = False

    def __len__(self):
        """Length of the dataset.
        """
        return len(self.chirps)

    def __getitem__(self, index):
        """Get item.
        """
        chirps = self.chirps[index]
        chirps = torch.tensor(chirps).unsqueeze(0)
        chirps_low = transforms.Resize((self.cmip6.shape[1], self.cmip6.shape[2]))(chirps)

        cmip6 = self.cmip6[index]
        cmip6 = torch.tensor(cmip6).unsqueeze(0)
        
        if self.transforms:
            images = torch.cat((cmip6, chirps_low), 0)
            images = self.transforms(images)
            cmip6 = images[0].unsqueeze(0)
            chirps_low = images[1].unsqueeze(0)

        return cmip6.to(torch.float32), chirps_low.to(torch.float32)


class ChirpsDataModule(pl.LightningDataModule):
    
        def __init__(self, data_dir="dataset/high-low", batch_size=32, transforms=False, scale=5, crop=160) -> None:
    
            super().__init__()
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.transforms = transforms
            self.scale = scale
            self.crop = crop
    
        def prepare_data(self):
            """Prepare data.
            """
            ...
    
        def setup(self, stage=None):
            """Setup data.
    
            Args:
                stage (str, optional): Stage. Defaults to None.
            """
            if stage == 'fit':
                self.train_dataset = Chirps(data_dir=self.data_dir, type='train', transformations=self.transforms,
                                            scale=self.scale, crop=self.crop)
                self.val_dataset = Chirps(data_dir=self.data_dir, type='val', transformations=False,
                                            scale=self.scale, crop=self.crop)
            if stage == 'test':
                self.test_dataset = Chirps(data_dir=self.data_dir, type='test', transformations=False,
                                            scale=self.scale, crop=self.crop)
    
        def train_dataloader(self):
            """Train dataloader.
            """
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8
            )
    
        def val_dataloader(self):
            """Validation dataloader.
            """
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8
            )
    
        def test_dataloader(self):
            """Test dataloader.
            """
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8
            )

class ChirpsCmip6DataModule(pl.LightningDataModule):

    def __init__(self, data_dir="dataset/high-low", batch_size=32, transforms=False) -> None:

        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms

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
            self.train_dataset = ChirpsCmip6(data_dir=self.data_dir, type='train', transformations=self.transforms)
            self.val_dataset = ChirpsCmip6(data_dir=self.data_dir, type='val', transformations=False)
        if stage == 'test' or stage is None:
            self.test_dataset = ChirpsCmip6(data_dir=self.data_dir, type='test', transformations=False)


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
    
class BlurChirpsDataModule(pl.LightningDataModule):
    
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
            if stage == 'fit':
                self.train_dataset = BlurChirps(data_dir=self.data_dir, type='train')
                self.val_dataset = BlurChirps(data_dir=self.data_dir, type='val')
            if stage == 'test':
                self.test_dataset = BlurChirps(data_dir=self.data_dir, type='test')
    
        def train_dataloader(self):
            """Train dataloader.
            """
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8
            )
    
        def val_dataloader(self):
            """Validation dataloader.
            """
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8
            )
    
        def test_dataloader(self):
            """Test dataloader.
            """
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8
            )