#!/bin/bash

# Destiny directory
dest_dir="dataset/CNRM-CMIP6"

# Create a destiny directory if it doesn't exist
mkdir -p "$dest_dir"

for year in {1981..2014}
do
    filename="pr_day_CNRM-CM6-1_historical_r1i1p1f2_gr_${year}.nc"
    filepath="${dest_dir}/${filename}"
    if [ ! -f "$filepath" ]
    then
        echo "Downloading $filename"
        wget "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6/CNRM-CM6-1/historical/r1i1p1f2/pr/$filename" -P $dest_dir
    else
        echo "File $filename already exists, skipping."
    fi
done

