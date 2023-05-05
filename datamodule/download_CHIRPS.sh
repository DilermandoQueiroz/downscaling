#!/bin/bash

# Destiny directory
dest_dir="dataset/CHIRPS"

# Create a destiny directory if it doesn't exist
mkdir -p "$dest_dir"
url="https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
wget -P "$dest_dir" "$url"
