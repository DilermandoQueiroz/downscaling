import lightning.pytorch as pl
import torch
import numpy as np
import xarray as xr
import pandas as pd
from torchvision import transforms
from copy import deepcopy
import random


class Geospatial(torch.utils.data.Dataset):

    def __init__(self, data_dir, type):
        self.data_dir = data_dir
        self.type = type

        if type == 'train':
            self.cmip6 = np.load(f'{data_dir}/train_cmip6.npy')
            self.chirps = np.load(f'{data_dir}/train_chirps.npy')
        elif type == 'val':
            self.cmip6 = np.load(f'{data_dir}/val_cmip6.npy')
            self.chirps = np.load(f'{data_dir}/val_chirps.npy')
        elif type == 'test' or 'eval':
            self.cmip6 = np.load(f'{data_dir}/test_cmip6.npy')
            self.chirps = np.load(f'{data_dir}/test_chirps.npy')
        else:
            raise ValueError(f'Invalid type: {type} (must be train, val or test)')

        self.chirps[np.isnan(self.chirps)] = 0
        self.cmip6[np.isnan(self.cmip6)] = 0

        # find mean and std
        self.chirps_max = self.chirps.max()
        self.chirps_min = self.chirps.min()
        self.cmip6_max = self.cmip6.max()
        self.cmip6_min = self.cmip6.min()

    @staticmethod
    def reconstruct_image(images: np.array, ratio=5):
        num_images, _, _ = images.shape
        images = images.reshape(num_images // (ratio * ratio), ratio * ratio, images.shape[1], images.shape[1])
        _, num_pieces, piece_size, _ = images.shape

        pieces_axis = int(np.sqrt(num_pieces))

        reconstruct_image_size = pieces_axis * piece_size
        reconstruct_images = []

        for img_set in images:
            reconstructed_image = np.zeros((reconstruct_image_size, reconstruct_image_size))

            i = 0
            for k in range(pieces_axis):
                for j in range(pieces_axis):
                    reconstruct_init_hor = piece_size * k
                    reconstruct_end_hor = piece_size * k + piece_size
                    reconstruct_init_vert = piece_size * j
                    reconstruct_end_vert = piece_size * j + piece_size
                    reconstructed_image[reconstruct_init_hor:reconstruct_end_hor, reconstruct_init_vert:reconstruct_end_vert] = img_set[i]
                    i += 1

            reconstruct_images.append(np.flip(reconstructed_image, axis=0))

        return np.array(reconstruct_images)
    
    @staticmethod
    def create_xarray(data: np.array, start_date: pd.Timestamp = pd.Timestamp('2007-01-01'), bbox=[-35, -75, 5, -35]):
        num_time_steps, image_size, _ = data.shape
        
        date_range = xr.cftime_range(start=start_date, periods=len(num_time_steps), freq="1M")

        # Create the time_coords using xr.DataArray with date values
        time_coords = xr.DataArray(date_range, dims=("time",), attrs={"units": "months"})

        # Create latitude and longitude arrays
        latitude_values = np.arange(bbox[0], bbox[2], image_size)
        longitude_values = np.arange(bbox[1], bbox[3], image_size)

        # Create coordinate arrays using xarray DataArray
        latitude_coords = xr.DataArray(latitude_values, dims=("lat",), attrs={"units": "degrees_north"})
        longitude_coords = xr.DataArray(longitude_values, dims=("lon",), attrs={"units": "degrees_east"})

        # Create a DataArray with the input data and coordinate values
        data_array = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time_coords, "lat": latitude_coords, "lon": longitude_coords})

        # Create the xarray Dataset
        dataset = xr.Dataset({"data": data_array})

        return dataset



class BlurChirps(Geospatial):
    """ BlurChirps dataset.
    """

    def __init__(self, data_dir="dataset/high-low", type='train', transformations=False, scale=5, crop=160) -> None:
        super().__init__(data_dir, type)
        
        # Transform nan to zero
        self.chirps[np.isnan(self.chirps)] = 0
        
        # Remove images that are all zero
        self.chirps = np.array([image for image in self.chirps if image.sum() != 0])
        self.chirps = (self.chirps - self.chirps_min) / (self.chirps_max - self.chirps_min)

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
    
class Chirps(Geospatial):
    """Chirps dataset.
    """

    def __init__(self, data_dir="dataset/high-low", type='train', transformations=False, scale=5, crop=160) -> None:
        super().__init__(data_dir, type)
        self.crop = crop
        self.scale = scale
        self.transformations = transformations

        if crop % scale != 0:
            raise ValueError(f'the scale need to be a factor of crop')
    
        self.image_size = self.chirps.shape[1]

        if crop > self.image_size:
            raise f"The crop size {crop} cannot be largen the image size {self.image_size}"
        
        if type != 'eval':
            # Remove images that are all zero
            self.chirps = np.array([image for image in self.chirps if image.sum() != 0])

        self.chirps = (self.chirps - self.chirps_min) / (self.chirps_max - self.chirps_min)
        
        if crop < self.image_size:
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
            
        elif self.transformations and crop == self.image_size:
            self.transform = transforms.Compose([
                transforms.Pad(32),
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
        # chirps_low = transforms.Resize((self.image_size, self.image_size), antialias=True)(chirps_low)

        if self.transformations:
            images = torch.cat((chirps_low.unsqueeze(0), chirps.unsqueeze(0)), 0)
            transformed_images = self.transform(images)
            chirps_low = transformed_images[0]
            chirps = transformed_images[1]

        return chirps_low.to(torch.float32), chirps.to(torch.float32)

class ChirpsCmip6(Geospatial):
    """ChirpsCmip6 dataset.
    """
    def __init__(self, data_dir="dataset/high-low", type='train', transformations=False) -> None:
        super().__init__(data_dir, type)

        # if chirps and cmip6 are all zero, remove it
        if type != 'eval':
            self.chirps2 = []
            self.cmip62 = []
            for i in range(len(self.chirps)):
                if self.chirps[i].sum() != 0 and self.cmip6[i].sum() != 0:
                    self.chirps2.append(self.chirps[i])
                    self.cmip62.append(self.cmip6[i])
        
            self.chirps = np.array(self.chirps2)
            self.cmip6 = np.array(self.cmip62)

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


class GeospatialBaseModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, transforms):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms

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

class ChirpsDataModule(GeospatialBaseModule):
    
        def __init__(self, data_dir="dataset/high-low", batch_size=32, transforms=False, scale=5, crop=160) -> None:
            super().__init__(data_dir, batch_size, transforms)
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.transforms = transforms
            self.scale = scale
            self.crop = crop
        
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

class ChirpsCmip6DataModule(GeospatialBaseModule):

    def __init__(self, data_dir="dataset/high-low", batch_size=32, transforms=False) -> None:
        super().__init__(data_dir, batch_size, transforms)

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

    
class BlurChirpsDataModule(GeospatialBaseModule):
    
        def __init__(self, data_dir="dataset/high-low", batch_size=32, transforms=False) -> None:
            super().__init__(data_dir, batch_size, transforms)
    
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