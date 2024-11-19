# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:44:02 2024

@author: PC2user
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 2024
@description: Global satellite data and in-situ data matching program
"""
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
import os
import glob
import h5py
from scipy.spatial import cKDTree
import logging
from datetime import datetime
import warnings
from itertools import groupby
from operator import itemgetter

def setup_logging(output_dir):
    """设置日志记录器"""
    log_file = os.path.join(output_dir, f'matching_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建文件处理器和流处理器
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, file_handler

def get_utm_zone(longitude, latitude):
    """根据经纬度确定UTM投影带号和南北半球"""
    utm_zone = int(((longitude + 180) / 6) % 60) + 1
    is_northern = latitude >= 0
    return utm_zone, is_northern

def get_utm_epsg(longitude, latitude):
    """根据经纬度获取对应的UTM投影EPSG代码"""
    utm_zone, is_northern = get_utm_zone(longitude, latitude)
    if is_northern:
        epsg = 32600 + utm_zone
    else:
        epsg = 32700 + utm_zone
    return epsg

def geomatching_nc(latitude, longitude, image_data, points_data, bands):
    """
    批量处理同一影像中的多个点位匹配
    
    Parameters:
    -----------
    latitude : 2D array
        卫星影像的纬度数组
    longitude : 2D array
        卫星影像的经度数组
    image_data : 3D array
        卫星影像数据 (bands, height, width)
    points_data : DataFrame
        包含待匹配点位的数据框
    bands : list
        波段名称列表
    """
    results = []
    empty_result = np.concatenate([np.array([0, 0]), np.array([np.nan] * len(bands))])
    
    # 数据有效性检查
    if np.any(np.isnan(latitude)) or np.any(np.isnan(longitude)):
        raise ValueError("Input latitude/longitude contains NaN values")
    
    # 计算影像中心点的位置
    center_lon = np.mean(longitude)
    center_lat = np.mean(latitude)
    
    # 获取第一个点的UTM投影
    utm_epsg = get_utm_epsg(points_data['Lon'].iloc[0], points_data['Lat'].iloc[0])
    
    # 建立投影转换器
    src_crs = CRS.from_epsg(4326)  # WGS84
    dst_crs = CRS.from_epsg(utm_epsg)
    latlon_to_utm = Transformer.from_proj(src_crs, dst_crs, always_xy=True)
    
    # 将卫星影像坐标转换为UTM
    sat_lon_utm, sat_lat_utm = latlon_to_utm.transform(longitude.flatten(), latitude.flatten())
    sat_coords = np.vstack((sat_lon_utm, sat_lat_utm)).T
    
    # 构建KD树
    tree = cKDTree(sat_coords)
    
    # 处理每个点位
    for _, point in points_data.iterrows():
        lat, lon, GLORIA_ID, Satellite_Name = point['Lat'], point['Lon'], point['Point'], point['Satellite_Name']
        point_information = np.array(point)  # 保存原始经纬度
        
        # 检查点是否在影像合理范围内
        if abs(center_lon - lon) > 5 or abs(center_lat - lat) > 5:
            logging.warning(f'Point ({lat}, {lon}) is far from image center ({center_lat}, {center_lon})')
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            continue
            
        # 转换目标点坐标
        lon_utm, lat_utm = latlon_to_utm.transform(lon, lat)
        
        # 设置距离阈值
        max_distance = 30  # meters
        dist, idx = tree.query([lon_utm, lat_utm])
        
        if dist >= max_distance:
            logging.warning(f'Point ({lat}, {lon}) is too far from any pixel center (distance: {dist:.2f}m)')
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            continue
        
        # 获取行列号
        pixel_row, pixel_col = np.unravel_index(idx, latitude.shape)
        
        # 检查点是否在影像范围内并处理
        if not (0 <= pixel_row < latitude.shape[0] and 0 <= pixel_col < latitude.shape[1]):
            logging.warning(f'Point ({lat}, {lon}) is outside image range')
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            continue
            
        logging.info(f'Point ({lat}, {lon}) matched at pixel ({pixel_row}, {pixel_col})')
        
        # 提取3x3窗口数据
        window = ((max(0, pixel_row - 1), min(latitude.shape[0], pixel_row + 2)),
                 (max(0, pixel_col - 1), min(latitude.shape[1], pixel_col + 2)))
        
        window_size = (window[0][1] - window[0][0]) * (window[1][1] - window[1][0])
        if window_size < 9:
            logging.warning(f'Incomplete window size: {window_size} pixels')
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            continue
        
        try:
            # 提取数据并进行质量控制
            data_box = image_data[1, window[0][0]:window[0][1], window[1][0]:window[1][1]]
            data_box = np.where(data_box == -10000, np.nan, data_box)
            valid_pixels = data_box[~np.isnan(data_box)]
            
            if not valid_pixels.size > 0:
                logging.warning(f'No valid pixels for point ({lat}, {lon})')
                results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
                continue
                
            valid_ratio = valid_pixels.size / data_box.size
            if not valid_ratio > 0.5:
                logging.warning(f'Valid pixel ratio ({valid_ratio:.2f}) < 0.5 for point ({lat}, {lon})')
                results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
                continue
                
            mean_value = np.mean(valid_pixels)
            if mean_value == 0:
                logging.warning(f'Mean value is zero for point ({lat}, {lon})')
                results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
                continue
                
            cv = np.std(valid_pixels) / mean_value
            if not cv < 0.15:
                logging.warning(f'CV ({cv:.3f}) > 0.15 for point ({lat}, {lon})')
                results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
                continue
                
            data_box_ = image_data[:, window[0][0]:window[0][1], window[1][0]:window[1][1]]
            band_means = np.nanmean(data_box_, axis=(1,2))
            
            quality_info = {
                'utm_zone': get_utm_zone(lon, lat)[0],
                'valid_pixel_ratio': valid_ratio,
                'cv': cv,
                'distance': dist,
                'window_size': window_size
            }
            logging.info(f'Quality metrics: {quality_info}')
            
            results.append(np.concatenate([point_information, band_means]))
            
        except Exception as e:
            logging.error(f"Error processing window data for point ({lat}, {lon}): {e}")
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            
    return results

def find_nearest_pixel(src, lon, lat, max_distance=0.01):
    """
    Find the nearest pixel to the given longitude and latitude.
    
    :param src: Rasterio dataset
    :param lon: Target longitude
    :param lat: Target latitude
    :param max_distance: Maximum allowed distance in degrees
    :return: Tuple of (px, py) or None if no pixel found within max_distance
    """
    # Get the coordinate reference system (CRS) of the image
    src_crs = src.crs

    # Create a transformer to convert between lat/lon and the image's CRS
    transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)

    # Transform the input lat/lon to the image's CRS
    x, y = transformer.transform(lon, lat)

    # Get the affine transform of the image
    affine = src.transform

    # Convert the transformed coordinates to pixel coordinates
    px, py = ~affine * (x, y)

    # Round to get the nearest pixel
    px, py = int(round(px)), int(round(py))

    # Check if the pixel is within the image bounds
    if 0 <= px < src.width and 0 <= py < src.height:
        # Get the coordinates of the found pixel
        found_x, found_y = affine * (px, py)
        found_lon, found_lat = transformer.transform(found_x, found_y, direction="INVERSE")

        # Calculate the distance
        distance = np.sqrt((lon - found_lon)**2 + (lat - found_lat)**2)

        if distance <= max_distance:
            return px, py
    else:
        print('the pixel is NOT within the image bounds')
    return None

def geomatching_tif(lat, lon, geotiff_paths, max_distance=0.001):
    import rasterio
    from rasterio.windows import Window
    
    results = []
    px, py = None, None
    
    with rasterio.open(geotiff_paths[0]) as src:
        try:
            px, py = find_nearest_pixel(src, lon, lat, max_distance)
            # 确保3*3窗口不会越界
            px_start = max(0, px - 1)
            py_start = max(0, py - 1)
            px_end = min(src.width, px + 2)
            py_end = min(src.height, py + 2)
            
            window = Window(px_start, py_start, px_end - px_start, py_end - py_start)
            
            data_box = src.read(1,window=window)
            valid_pixels = data_box[~np.isnan(data_box)]
            if not valid_pixels.size > 0:
                logging.warning(f'No valid pixels for point ({lat}, {lon})')
                results.append(None)
            else:                
                valid_ratio = valid_pixels.size / data_box.size
                
            if not valid_ratio > 0.5:
                logging.warning(f'Valid pixel ratio ({valid_ratio:.2f}) < 0.5 for point ({lat}, {lon})')
                results.append(None)
            else:                
                mean_value = np.mean(valid_pixels)
                
            if mean_value == 0:
                logging.warning(f'Mean value is zero for point ({lat}, {lon})')
                results.append(None)
            else:                
                cv = np.std(valid_pixels) / mean_value
            if not cv < 0.15:
                logging.warning(f'CV ({cv:.3f}) > 0.15 for point ({lat}, {lon})')
                results.append(None)
            else:
                for path in geotiff_paths:
                    with rasterio.open(path) as src:
                        
                        data = src.read(1, window=window)
                        data = data*1.0
                        data[data==-9999] = np.nan
                        data = data / 100000
                        mean_value = np.nanmean(data)
                        results.append(mean_value)
        except ValueError as e:
            print(f"Error: {str(e)}")
            return [None] * len(geotiff_paths)
    
    return results

def process_data(path_df, path_image, ac_method):
    """处理多个点位和影像的匹配"""
    df = pd.read_csv(path_df)
    results = []
    
    # 按卫星名称分组处理
    df_sorted = df.sort_values('Satellite_Name')
    grouped_data = df_sorted.groupby('Satellite_Name')
    
    if ac_method == "acolite":
        # bands = ['Rrs_443', 'Rrs_482', 'Rrs_561', 'Rrs_655', 'Rrs_865']
        
        for satellite_name, group_df in grouped_data:
            if 'LC08' in satellite_name:
                bands = ['Rrs_443', 'Rrs_483', 'Rrs_561', 'Rrs_592', 'Rrs_613', 'Rrs_655', 'Rrs_865', 'Rrs_1609', 'Rrs_2201']
            
            elif 'LC09' in satellite_name:
                bands = ['Rrs_443', 'Rrs_482', 'Rrs_561', 'Rrs_594', 'Rrs_613', 'Rrs_654', 'Rrs_865', 'Rrs_1608', 'Rrs_2201']
                
            elif 'S2A' in satellite_name:
                bands = ['Rrs_443', 'Rrs_492', 'Rrs_560',  'Rrs_665', 'Rrs_704', 'Rrs_740', 'Rrs_783', 'Rrs_833', 'Rrs_865', 'Rrs_1614', 'Rrs_2202']
            
            elif 'S2B' in satellite_name:
                bands = ['Rrs_442', 'Rrs_492', 'Rrs_559',  'Rrs_665', 'Rrs_704', 'Rrs_739', 'Rrs_780', 'Rrs_833', 'Rrs_864', 'Rrs_1610', 'Rrs_2186']
            
            else:
                print("Sensor cannot be identified!")
                continue
                
            try:
                # acolite_file = glob.glob(os.path.join(path_image, 
                #                        satellite_name.replace('L1', 'L2'), 
                #                        '*L2W.nc'))
                acolite_file = glob.glob(os.path.join(path_image, 
                                       satellite_name.replace('L1', 'L2'), 
                                       '*L2W.nc'))
                
                if acolite_file:
                    logging.info(f"\nProcessing {satellite_name} with {len(group_df)} points...")
                    with h5py.File(acolite_file[0], 'r') as ds:
                        latitude = np.array(ds['lat'])
                        longitude = np.array(ds['lon'])
                        image_data = np.array([np.array(ds[band]) for band in bands])
                        
                        batch_results = geomatching_nc(latitude, longitude, image_data, 
                                                  group_df, bands)
                        results.extend(batch_results)
                else:
                    logging.warning(f"No file found for {satellite_name}")
                    # 文件未找到时，为该组所有点添加空值结果
                    for _, point in group_df.iterrows():
                        results.append(np.concatenate([
                            np.array(point.tolist()),
                            np.array([np.nan] * len(bands))
                        ]))
                    
            except Exception as e:
                logging.error(f"Error processing {satellite_name}: {e}")
                # 处理出错时，为该组所有点添加空值结果
                for _, point in group_df.iterrows():
                    results.append(np.concatenate([
                        np.array([point['Lat'], point['Lon'], point['Point'], point['Satellite_Name']]),
                        np.array([np.nan] * len(bands))
                    ]))
                continue
                
    elif ac_method == "ocsmart":
        bands = ['Rrs_443nm', 'Rrs_482nm', 'Rrs_561nm', 'Rrs_655nm']
        
        for satellite_name, group_df in grouped_data:
            try:
                ocsmart_file = os.path.join(path_image, f"{satellite_name}_L2_OCSMART.h5")
                
                if os.path.exists(ocsmart_file):
                    logging.info(f"\nProcessing {satellite_name} with {len(group_df)} points...")
                    with h5py.File(ocsmart_file, 'r') as ds:
                        latitude = np.array(ds['Latitude'])
                        longitude = np.array(ds['Longitude'])
                        image_data = np.array([np.array(ds[f'Rrs/{band}']) for band in bands])
                        
                        batch_results = geomatching_nc(latitude, longitude, image_data, 
                                                  group_df, bands)
                        results.extend(batch_results)
                else:
                    logging.warning(f"No file found for {satellite_name}")
                    # 文件未找到时，为该组所有点添加空值结果
                    for _, point in group_df.iterrows():
                        results.append(np.concatenate([
                            np.array([point['Latitude'], point['Longitude']]),
                            np.array([np.nan] * len(bands))
                        ]))
                    
            except Exception as e:
                logging.error(f"Error processing {satellite_name}: {e}")
                # 处理出错时，为该组所有点添加空值结果
                for _, point in group_df.iterrows():
                    results.append(np.concatenate([
                        np.array([point['Latitude'], point['Longitude'], point['GLORIA_ID'], point['Satellite_Name']]),
                        np.array([np.nan] * len(bands))
                    ]))
                continue
    
    elif ac_method == "seadas":
        bands = ['AR_BAND1', 'AR_BAND2', 'AR_BAND3', 'AR_BAND4', 'AR_BAND5']
        
        for index, row in df_sorted.iterrows():
            gloria_id = row['GLORIA_ID']
            satellite_name = row['Satellite_Name']
            latitude = row['Latitude']
            longitude = row['Longitude']
        
            mid_folder = satellite_name[:4] + satellite_name[10:16] + satellite_name[17:25]
            mid_path = glob.glob(os.path.join(path_image,mid_folder+"*"))
            
            
            # logging.info(f"\nProcessing {satellite_name} with {len(group_df)} points...")
            
            geotiff_paths = [os.path.join(mid_path[0],f"{satellite_name}_AR_BAND{i}.tif") for i in range(1, 6)]
            
            # Check if all files exist
            if all(os.path.exists(path) for path in geotiff_paths):
                # Get pixel values
                pixel_values = geomatching_tif(latitude, longitude, geotiff_paths)
                
                # Prepare row data
                row_data = {
                    'GLORIA_ID': gloria_id,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'Satellite_Name': satellite_name
                }
                
                # Add satellite band values
                for i, value in enumerate(pixel_values, 1):
                    row_data[f'AR_Band_{i}'] = value
                
                # Add spectral reflectance data
                # for column in spectral_df.columns:
                #     if column not in ['GLORIA_ID', 'latitude', 'longitude']:
                #         row_data[column] = row[column]
                
                results.append(row_data)
            else:
                print(f"Skipping {gloria_id}: Some satellite data files are missing")
            
    # 创建结果DataFrame
    if results:
        columns = df.columns.tolist() + bands
        if ac_method == "seadas":
            df_result = pd.DataFrame(results)
        else:
            df_result = pd.DataFrame(results, columns=columns)
        return df_result
    else:
        logging.warning("No results found")
        # 返回空DataFrame但保持列结构
        return pd.DataFrame(columns=df.columns.tolist() + bands)

def main():
    """主函数：处理匹配过程并保存结果"""
    # 配置参数
    config = {
        'ac_method': "acolite",  # 可选: "acolite" 或 "ocsmart" or "seadas"
        'path_df': r'C:\Users\jmen\Box\Database_Africa\Africa_water_quality_dataset-2023.csv',
        'path_image': r'F:\all_satellite_L2',
        'output_dir': r'C:\Users\jmen\Box\Database_Africa\results'
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 设置日志
    logger, file_handler = setup_logging(config['output_dir'])
    
    try:
        # 记录开始时间
        start_time = datetime.now()
        logger.info(f"Processing started at {start_time}")
        logger.info(f"Using atmospheric correction method: {config['ac_method']}")
        
        # 处理数据
        results_df = process_data(config['path_df'], config['path_image'], config['ac_method'])
        
        if results_df is not None and not results_df.empty:
            # 生成输出文件名
            output_filename = f"matchups_{config['ac_method']}_{start_time.strftime('%Y%m%dT%H%M%S')}.csv"
            output_path = os.path.join(config['output_dir'], output_filename)
            
            # 保存结果
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Total matched points: {len(results_df)}")
            
            # 生成匹配统计信息
            stats = {
                'total_input_points': len(pd.read_csv(config['path_df'])),
                'successful_matches': len(results_df),
                'match_rate': len(results_df) / len(pd.read_csv(config['path_df'])) * 100
            }
            
            logger.info("Matching Statistics:")
            logger.info(f"Total input points: {stats['total_input_points']}")
            logger.info(f"Successful matches: {stats['successful_matches']}")
            logger.info(f"Match rate: {stats['match_rate']:.2f}%")
            
            # 保存统计信息
            stats_file = os.path.join(config['output_dir'], 
                                    f"matching_statistics_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")
            with open(stats_file, 'w') as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
            
        else:
            logger.warning("No valid matches found in the data")
            
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        raise
        
    finally:
        # 记录结束时间和总运行时间
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Processing completed at {end_time}")
        logger.info(f"Total processing time: {duration}")
        
        # 关闭文件处理器
        logger.removeHandler(file_handler)
        file_handler.close()

if __name__ == '__main__':
    # 设置警告处理
    warnings.filterwarnings('always')  # 或者使用 'ignore' 来抑制警告
    
    try:
        main()
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error occurred in main program: {str(e)}")
        raise