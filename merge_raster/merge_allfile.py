# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:51:07 2024

@author: jmen
"""
import os.path
import numpy as np
import glob
import rasterio
from rasterio.merge import merge

def copy_mean(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    mask_merged_zero = merged_data==0.0
    mask_new_zero = new_data==0.0
    np.logical_or(mask_new_zero, new_mask, out=new_mask) 
    np.logical_or(mask_merged_zero, merged_mask, out=merged_mask) 
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    
    np.add(merged_data,new_data,out=merged_data,where=mask)
    np.true_divide(merged_data,2,out=merged_data,where=mask)
    
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")
    
def copy_frequence(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    mask_merged_zero = merged_data==0.0
    mask_new_zero = new_data==0.0
    np.logical_or(mask_new_zero, new_mask, out=new_mask) 
    '''将new_data 的有效像元标记为1'''
    new_data.data[new_data.data>0] = 1
    new_data.data[new_data.data<0] = 1
    # np.logical_not(new_mask, out=new_data.data)
    '''求merged_data和new_data重叠部分'''
    np.logical_or(mask_merged_zero, merged_mask, out=merged_mask) 
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    
    np.add(merged_data,new_data,out=merged_data,where=mask)
    
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")
    
def merge_raster_images_tiff(fnames,outname):
    '''
    The function for merging raster images with .tif format.
    :param fnames: raster images.
    :param outname: target name.
    :return: no return.
    '''

    fmodels = []
    for fname in fnames:
        src = rasterio.open(fname)
        fmodels.append(src)

    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(fmodels,method=copy_mean) #"max","min"

    # Copy the metadata
    out_meta = src.meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": src.crs, # "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                     "compress": 'lzw',
                     # "dtype": 'uint8'
                    })
    mosaic = np.array(mosaic,dtype='float32')
    with rasterio.open(outname, "w", **out_meta) as dest:
        dest.write(mosaic)

    print('Finished!')
    
if __name__=='__main__':
    path = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeNicol\2024Oct22_LakeNicol\Micasense\Merge'
    outname = os.path.join(path,'LakeNicol_20241022_merge.tif')
    
    fnames = glob.glob(os.path.join(path,'*.tif'))
    
    merge_raster_images_tiff(fnames, outname)
