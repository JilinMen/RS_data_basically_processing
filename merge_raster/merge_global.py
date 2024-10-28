# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:44:14 2022

@author: Administrator
"""
import os.path
import numpy as np
import glob
import rasterio
from rasterio.merge import merge
import multiprocessing
import datetime
# import numpy.ma as ma

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

def CI(band1,band2,band3):
    '''
    MODIS
    band1:443
    band2:555
    band3:667
    '''
    ci = band2-0.5*(band1+band3)
    ci[np.argwhere(ci>0.004)] = 0.004   #ci指数小于-0.0005的可用于计算chl, 大概范围【-0.008，0.004】
    # chl = 10**(-0.4909+191.6590*ci)   #OCI 1
    chl = 10**(-0.4287+230.47*ci)       #OCI 2
    return chl.squeeze()

def out_date_by_day(year,day):
    '''
    根据输入的年份和天数计算对应的日期
    '''
    first_day=datetime.datetime(year,1,1)
    add_day=datetime.timedelta(days=day-1)
    return datetime.datetime.strftime(first_day+add_day,"%Y.%m.%d")

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
    mosaic, out_trans = merge(fmodels,method=copy_frequence)

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

def merge_daily_timeseries():
    
    fdir = r'Z:\AC\MOD_L2_201206_global'
    outdir = r'Z:\AC\MOD_L2_201206_global\composite'
    year = '2012'
    # fnames = []
    # for root, dirs, files in os.walk(fdir):
    #     for file in files:
    #         if file[-10:] == 'sd_CHL.tif':
    #             fnames.append(os.path.join(root, file))
    month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    '''SeaDAS data'''
    for i in month:
        print('SeaDAS data process '+i+'/'+str(len(month)))
        fnames = glob.glob(os.path.join(fdir,'A'+year+'1'+str(int(i)+50)+'*sd_CHL.tif'))
        outname = os.path.join(outdir, 'A'+year+'1'+str(int(i)+50)+'_daily_SD.tif')
        if os.path.exists(outname):
            print(outname,'existed')
        elif len(fnames)>0:
                merge_raster_images_tiff(fnames, outname)
    '''DL data'''
    for i in month:
        print('DL data process '+i+'/'+str(len(month)))
        fnames = glob.glob(os.path.join(fdir,'A'+year+'1'+str(int(i)+50)+'*dl_CHL.tif'))
        outname = os.path.join(outdir, 'A'+year+'1'+str(int(i)+50)+'_daily_DL.tif')
        if os.path.exists(outname):
            print(outname,'existed')
        elif len(fnames)>0:
                merge_raster_images_tiff(fnames, outname)

def merge_globalRrs_daily():
    '''
    band merge
    '''
    fdir = r'Z:\AC\MOD_L2_200306_global'
    outdir = r'Z:\AC\MOD_L2_200306_global\global_Rrs'
    year = '2003'
    bands = ['443','547','667']
    days = ['151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163',
               '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176',
               '177', '178', '179', '180', '181', '182']
    # fnames = []
    # for root, dirs, files in os.walk(fdir):
    #     for file in files:
    #         if file[-10:] == 'sd_CHL.tif':
    #             fnames.append(os.path.join(root, file))
    for band in bands:
        for day in days:
            '''DL data'''
            fnames = glob.glob(os.path.join(fdir,'A'+year+day+'*dl_'+band+'_Rrs.tif'))
            outname = os.path.join(outdir, 'A'+year+day+'_daily_'+band+'_DL.tif')
            if os.path.exists(outname):
                print(outname,'existed')
            elif len(fnames)>0:
                    merge_raster_images_tiff(fnames, outname)
            
            '''SeaDAS'''
            fnames = glob.glob(os.path.join(fdir,'A'+year+day+'*sd_'+band+'_Rrs.tif'))
            outname = os.path.join(outdir, 'A'+year+day+'_daily_'+band+'_SD.tif')
            if os.path.exists(outname):
                print(outname,'existed')
            elif len(fnames)>0:
                    merge_raster_images_tiff(fnames, outname)
                    
def merge_globalRrs_monthly():
    '''
    band merge
    '''
    fdir = r'Z:\AC\MOD_L2_200306_global\global_Rrs'
    outdir = r'Z:\AC\MOD_L2_200306_global\global_Rrs'
    year = '2003'
    bands = ['443','547','667']
    # fnames = []
    # for root, dirs, files in os.walk(fdir):
    #     for file in files:
    #         if file[-10:] == 'sd_CHL.tif':
    #             fnames.append(os.path.join(root, file))
    for band in bands:
        '''DL data'''
        fnames = glob.glob(os.path.join(fdir,'A'+year+'*_daily_'+band+'_DL.tif'))
        outname = os.path.join(outdir, 'A'+year+'06_mean_monthly_'+band+'_DL.tif')
        if os.path.exists(outname):
            print(outname,'existed')
        elif len(fnames)>0:
                merge_raster_images_tiff(fnames, outname)
        
        '''SeaDAS'''
        fnames = glob.glob(os.path.join(fdir,'A'+year+'*_daily_'+band+'_SD.tif'))
        outname = os.path.join(outdir, 'A'+year+'06_mean_monthly_'+band+'_SD.tif')
        if os.path.exists(outname):
            print(outname,'existed')
        elif len(fnames)>0:
                merge_raster_images_tiff(fnames, outname)

def merge_globalChl_monthly():
    '''
    band merge
    '''
    fdir = r'Z:\AC\MOD_L2_200306_global\composite'
    outdir = r'Z:\AC\MOD_L2_200306_global\composite'
    year = '2003'
    # fnames = []
    # for root, dirs, files in os.walk(fdir):
    #     for file in files:
    #         if file[-10:] == 'sd_CHL.tif':
    #             fnames.append(os.path.join(root, file))

    '''DL data'''
    fnames = glob.glob(os.path.join(fdir,'A'+year+'*_daily_DL.tif'))
    outname = os.path.join(outdir, 'A'+year+'06_monthly_DL.tif')
    merge_raster_images_tiff(fnames, outname)
    
    '''SeaDAS'''
    fnames = glob.glob(os.path.join(fdir,'A'+year+'*_daily_SD.tif'))
    outname = os.path.join(outdir, 'A'+year+'06_monthly_SD.tif')
    merge_raster_images_tiff(fnames, outname)
    
def merge_globalfre_daily():
    '''
    band merge
    '''
    fdir = r'Z:\AC\MOD_L2_200306_global'
    outdir = r'Z:\AC\valid obs&difference\global_200306'
    year = '2003'
    bands = ['443']
    # days = ['001',  '002',  '003',  '004',  '005',  '006',  '007',  '008',  
    #         '009', '010', '011', '012', '013', '014', '015', '016',
    #         '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
    #         '027', '028', '029', '030', '031']
    days = ['151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163',
                '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176',
                '177', '178', '179', '180', '181', '182']
    # fnames = []
    # for root, dirs, files in os.walk(fdir):
    #     for file in files:
    #         if file[-10:] == 'sd_CHL.tif':
    #             fnames.append(os.path.join(root, file))
    for band in bands:
        for day in days:
            '''DL data'''
            fnames = glob.glob(os.path.join(fdir,'A'+year+day+'*dl_'+band+'_Rrs.tif'))
            outname = os.path.join(outdir, 'A'+year+day+'_daily_'+band+'_DL.tif')
            if os.path.exists(outname):
                print(outname,'existed')
            elif len(fnames)>0:
                    merge_raster_images_tiff(fnames, outname)
            
            '''SeaDAS'''
            fnames = glob.glob(os.path.join(fdir,'A'+year+day+'*sd_'+band+'_Rrs.tif'))
            outname = os.path.join(outdir, 'A'+year+day+'_daily_'+band+'_SD.tif')
            if os.path.exists(outname):
                print(outname,'existed')
            elif len(fnames)>0:
                    merge_raster_images_tiff(fnames, outname)
                    
def merge_globalfre_monthly():
    '''
    band merge
    '''
    fdir = r'Z:\AC\valid obs&difference\global_200306'
    outdir = r'Z:\AC\valid obs&difference\global_200306'
    year = '2003'
    bands = ['443']
    # fnames = []
    # for root, dirs, files in os.walk(fdir):
    #     for file in files:
    #         if file[-10:] == 'sd_CHL.tif':
    #             fnames.append(os.path.join(root, file))
    for band in bands:
        '''DL data'''
        fnames = glob.glob(os.path.join(fdir,'A'+year+'*_daily_'+band+'_DL.tif'))
        outname = os.path.join(outdir, 'A'+year+'01_mean_monthly_'+band+'_DL.tif')
        if os.path.exists(outname):
            print(outname,'existed')
        elif len(fnames)>0:
                merge_raster_images_tiff(fnames, outname)
        
        '''SeaDAS'''
        fnames = glob.glob(os.path.join(fdir,'A'+year+'*_daily_'+band+'_SD.tif'))
        outname = os.path.join(outdir, 'A'+year+'01_mean_monthly_'+band+'_SD.tif')
        if os.path.exists(outname):
            print(outname,'existed')
        elif len(fnames)>0:
                merge_raster_images_tiff(fnames, outname)    
if __name__ == "__main__":
    
    # p = multiprocessing.Pool(16) #并行任务数
    # b = p.map(merge_globalRrs_daily())
    # p.close()
    # p.join()
    merge_globalfre_daily()
    merge_globalfre_monthly()
