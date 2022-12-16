#!/bin/sh

''' 
This script intends to cut the plots from the orthomosaic by using already designed shapefiles and a singularity image.
A docker image containing the gdal library should also work fine.
The shapefiles and the original orthomosaic.tif must be in the same directory.

Note:
This script can run multiple orthomosaic and shapefile pairs, however they need to have matching names.
Modify the script to make sure it will be able to find the files.
'''

# Download the image from Docker Hub if not already available
#singularity pull --dir $MYSCRATCH/singularity docker://osgeo/gdal:alpine-small-latest

# Loading the singularity module is required when working with the Pawsey supercomputer
module load singularity

# add the image path to a variable
singularity_image=$MYGROUP/singularity/gdal_alpine-small-latest.sif
orthomosaic_dir=$MYSCRATCH/weed_data/orthomosaics
shapefile_dir=$MYSCRATCH/shapefiles
out_path=$MYSCRATCH/weed_data/plots

# Run this code to clip the plots from the orthomosaic
for image in $(ls $orthomosaic_dir) ;
        do dir_name=$(basename ${image} _mosaic.tif)
        mkdir ${out_path}/${dir_name}
        for plot in $(ls ${shapefiles_dir}/${dir_name}/*shp) ;
                do fname=$(basename $plot .shp)
                singularity exec $singularity_image gdalwarp -srcnodata -32767 \
                -cutline $plot -crop_to_cutline -of GTiff  -dstnodata None \
                $image ${out_path}/$dir_name/${fname}.tif
                done
        done

