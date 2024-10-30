# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:11:09 2024

@author: jmen
"""
import pandas as pd
import numpy as np
import rasterio
import pyproj

def geomatching(path_df,path_image,path_out,head_name=None):
    # 加载实测点数据
    df = pd.read_csv(path_df)
    
    # 创建一个空列表，用于存储匹配结果
    results = []
    
    # 定义坐标系
    utm_crs = pyproj.CRS.from_epsg(32616)  # WGS 84 / UTM zone 16N
    wgs84_crs = pyproj.CRS.from_epsg(4326)  # WGS 84
    # 创建坐标转换器
    transformer = pyproj.Transformer.from_crs(wgs84_crs, utm_crs)
    
    # 加载卫星影像数据
    with rasterio.open(path_image) as src:
        # 获取影像的仿射变换参数
        transform = src.transform
        # 获取影像的波段数
        num_bands = src.count
    
        # 遍历每个实测点
        for index, row in df.iterrows():
            # 获取实测点的经纬度
            lon, lat = row['Lon'], row['Lat']
            
            print('Lon: %f; lat: %f'%(lon,lat))
        
            # 将经纬度转换为UTM坐标
            utm_easting, utm_northing = transformer.transform(lat, lon)  # 注意纬度在前
            print(f"UTM Easting: {utm_easting}; UTM Northing: {utm_northing}")
        
            # 将UTM坐标转换为图像坐标
            pixel_row, pixel_col = ~transform * (utm_easting, utm_northing)
            pixel_row, pixel_col = int(pixel_row), int(pixel_col)  # 转换为整数
            print(f"x: {pixel_row}; y: {pixel_col}")
            
            # 判断实测点是否在影像范围内
            if 0 <= pixel_row < src.height and 0 <= pixel_col < src.width:
                print('--This point is in the image range!')
                
                # 读取以该像元为中心的3x3的box内的像素值
                window = ((pixel_row - 1, pixel_row + 2), (pixel_col - 1, pixel_col + 2))
                
                try:
                    data_B3 = src.read(3, window=window)
                    
                    data_B3 = np.where(data_B3==-10000, np.nan, data_B3)
                    # 计算有效像元数量和变异系数
                    valid_pixels = data_B3[~np.isnan(data_B3)]
                    if valid_pixels.size > data_B3.size * 0.5:
                        print('---Valid pixels > 0.5!')
                        cv = np.std(valid_pixels) / np.mean(valid_pixels)
                        if cv < 0.15:
                            print('----CV < 0.15!')
                            data = src.read(window=window)
                            data = np.where(data==-10000, np.nan, data)
                            # 计算box内所有有效值的平均值
                            mean_value = np.nanmean(data,axis=(1,2))
                            # 将结果添加到列表中
                            results.append(np.concatenate((row.values,mean_value)))
                except Exception as e:
                    print(f"Error reading window data: {e}")
            else:
                print('--This point is NOT in the image range!')
    
    # 将结果保存为csv文件
    if head_name == None:
        df_result = pd.DataFrame(results)
    else:
        df_result = pd.DataFrame(results, columns=head_name)
    df_result.to_csv(path_out, index=False)
    
    print('Save csv to %s'%(path_out))

if __name__=='__main__':
    # input csv file
    path_df = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeNicol\2024Oct22_LakeNicol\ASD\Rrs\LakeNicol_20241022_Rrs.csv'
    # input satellite/drone .tif image
    path_image = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeNicol\2024Oct22_LakeNicol\Micasense\Merge\LakeNicol_20241022_merge.tif'
    # output path
    path_out = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeNicol\2024Oct22_LakeNicol\Matchups\ASD-Micasense\ASD_Micasense_20241022.csv'
    # head name for each column
    head_name = ['FID','Id','NTU','Chl','pH','fDOM','DO','NitraLED','SpCond','Lat','Lon',\
                 'AR_B1','AR_B2','AR_B3','AR_B4','AR_B5','AR_B6','AR_B7','AR_B8','AR_B9','AR_B10']
        
    geomatching(path_df, path_image, path_out)