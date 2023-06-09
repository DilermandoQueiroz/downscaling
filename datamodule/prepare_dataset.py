from typing import List
import xarray as xr
import numpy as np
import argparse
from rasterio.enums import Resampling

def prepare_data(data_source: str, data_dir: str, time_init: str, time_end: str, bbox: List[float]):
    """Prepare CMIP6 data for the given time period and bounding box.

    Args:
        data_source (str): Data source.
        data_dir (str): Path to the CMIP6 data.
        time_init (int): Initial time in the format YYYYMMDD.
        time_end (int): End time in the format YYYYMMDD.
        bbox (List[float, float, float, float]): Bounding box in the format [lat_min, lon_min, lat_max, lon_max].

    Returns:
        data (xarray.Dataset): CMIP6 data for the given time period and bounding box.
    """
    if data_source == "cmip6":
        data = xr.open_mfdataset(f'{data_dir}/*.nc', combine='by_coords')
        data = data * 86400
        data = data.resample(time='1MS').sum()
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
        data = data.roll(lon=int(len(data['lon']) / 2), roll_coords=True)
        data = data.sel(time=slice(time_init, time_end), lat=slice(bbox[0], bbox[2]), lon=slice(bbox[1], bbox[3]))
        data = data.rename({'lat': 'latitude', 'lon': 'longitude'})

    elif data_source == "chirps" or data_source == "low_chirps":
        data = xr.open_dataset(data_dir)
        data = data.sel(time=slice(time_init, time_end), latitude=slice(bbox[0], bbox[2]), longitude=slice(bbox[1], bbox[3]))
        data = data.rename({'precip': 'pr'})

        if data_source == "low_chirps":
            data = data.rio.write_crs("EPSG:4326")
            
            upscale_factor = 5

            new_width = data.rio.width // upscale_factor
            new_height = data.rio.height // upscale_factor

            data = data.rio.reproject(
                data.rio.crs,
                shape=(new_height, new_width),
                resampling=Resampling.average,
            )
            data = data.rename({'y': 'latitude', 'x': 'longitude'})
    
    return data

def slicing(data: xr.Dataset, size: int):
    """Slicing the data into smaller pieces and convert to numpy arrays.

    Args:
        data (xarray.Dataset): CMIP6 data for the given time period and bounding box.
        size (int): Size of the slices.

    Returns:
        data_slices (List[float, float]): List of slices.
    """
    data_slices = []
    for year in range(0, len(data['time'])):
        for latitude in range(0, len(data['latitude']), size):
            for longitude in range(0, len(data['longitude']), size):
                data_slices.append(data.pr.isel(time=year, latitude=slice(latitude, latitude+size), longitude=slice(longitude, longitude+size)).values)
    
    return np.array(data_slices)

def main(data_dir: str, time_init: str, time_end: str, bbox: List[float], size: int, data_dir_save: str, type: str):
    print('PREPARING CMIP6 DATA -------')
    cmip6 = prepare_data(data_source="cmip6", data_dir=f"{data_dir}/CNRM-CMIP6",
                          time_init=time_init, time_end=time_end, bbox=bbox)

    print('PREPARING LOW RESOLUTION CHIRPS DATA')
    low_chirps = prepare_data(data_source="low_chirps", data_dir=f"{data_dir}/CHIRPS/chirps-v2.0.monthly.nc",
                          time_init=time_init, time_end=time_end, bbox=bbox)
    
    print('PREPARING CHIRPS DATA -------')
    chirps = prepare_data(data_source="chirps", data_dir=f"{data_dir}/CHIRPS/chirps-v2.0.monthly.nc",
                          time_init=time_init, time_end=time_end, bbox=bbox) 
    ratio = len(chirps.longitude) / len(cmip6.longitude)

    print('SLICING LOW RESOLUTION CHIRPS DATA -------')
    np_low_chirps = slicing(low_chirps, size)
    np.save(f'{data_dir_save}/{type}_low_chirps.npy', np_low_chirps)

    print('SLICING CHIRPS DATA -------')
    np_chirps = slicing(chirps, int(size * ratio))
    np.save(f'{data_dir_save}/{type}_chirps.npy', np_chirps)
    
    print('SLICING CMIP6 DATA -------')
    np_cmip6 = slicing(cmip6, size)
    np.save(f'{data_dir_save}/{type}_cmip6.npy', np_cmip6)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare CMIP6/CHIRPS data for the given time period and bounding box.')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to the data.')
    parser.add_argument('--data_dir_save', type=str, default='dataset/high-low', help='Local to save.')
    parser.add_argument('--type', type=str, default='train', help='Train data.')
    parser.add_argument('--time_init', type=str, default='1981', help='Initial time in the format YYYY.')
    parser.add_argument('--time_end', type=str, default='2014', help='End time in the format YYYY.')
    parser.add_argument('--bbox', type=List[float], default=[-35, -75, 5, -35], help='Bounding box in the format [lat_min, lon_min, lat_max, lon_max].')
    parser.add_argument('--size', type=int, default=32, help='Size of the slices.')

    args = parser.parse_args()
    with open(f'{args.data_dir_save}/{args.type}_args.txt', 'w') as f:
        f.write(str(args))
    
    main(**vars(parser.parse_args()))
