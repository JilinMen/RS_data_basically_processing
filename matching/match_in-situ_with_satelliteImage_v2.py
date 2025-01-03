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

    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, file_handler

def get_utm_zone(longitude, latitude):
    """Determine the UTM zone number and hemisphere based on latitude and longitude."""
    utm_zone = int(((longitude + 180) / 6) % 60) + 1
    is_northern = latitude >= 0
    return utm_zone, is_northern

def get_utm_epsg(longitude, latitude):
    """calculate EPSG from lat and lon-经纬度获取对应的UTM投影EPSG代码"""
    utm_zone, is_northern = get_utm_zone(longitude, latitude)
    if is_northern:
        epsg = 32600 + utm_zone
    else:
        epsg = 32700 + utm_zone
    return epsg

def geomatching_nc(latitude, longitude, image_data, points_data, bands):
    """
    matching multiple points with one image
    
    Parameters:
    -----------
    latitude : 2D array
        
    longitude : 2D array
        
    image_data : 3D array
        satellite image (bands, height, width)
    points_data : DataFrame
        in situ data
    bands : list
        band name list
    """
    results = []
    empty_result = np.concatenate([np.array([0, 0]), np.array([np.nan] * len(bands))])
    
    # check if data is valid
    if np.any(np.isnan(latitude)) or np.any(np.isnan(longitude)):
        raise ValueError("Input latitude/longitude contains NaN values")
    
    # calculate the lat and lon for center point of the image
    center_lon = np.mean(longitude)
    center_lat = np.mean(latitude)
    
    # get the UTM information from the first pixels
    utm_epsg = get_utm_epsg(points_data['Lon'].iloc[0], points_data['Lat'].iloc[0])
    
    # create projection transform
    src_crs = CRS.from_epsg(4326)  # WGS84
    dst_crs = CRS.from_epsg(utm_epsg)
    latlon_to_utm = Transformer.from_proj(src_crs, dst_crs, always_xy=True)
    
    # project geolocations to UTM
    sat_lon_utm, sat_lat_utm = latlon_to_utm.transform(longitude.flatten(), latitude.flatten())
    sat_coords = np.vstack((sat_lon_utm, sat_lat_utm)).T
    
    # build KD tree
    tree = cKDTree(sat_coords)
    
    # processing each in situ points
    for _, point in points_data.iterrows():
        lat, lon = point['Lat'], point['Lon']
        point_information = np.array(point)  # initial lat and lon
        
        # check if the in situ points are in the range of image
        if abs(center_lon - lon) > 5 or abs(center_lat - lat) > 5:
            logging.warning(f'Point ({lat}, {lon}) is far from image center ({center_lat}, {center_lon})')
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            continue
            
        # projection transform
        lon_utm, lat_utm = latlon_to_utm.transform(lon, lat)
        
        # spatial distance threshold
        max_distance = 30  # meters
        dist, idx = tree.query([lon_utm, lat_utm])
        
        if dist >= max_distance:
            logging.warning(f'Point ({lat}, {lon}) is too far from any pixel center (distance: {dist:.2f}m)')
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            continue
        
        # get row and col 
        pixel_row, pixel_col = np.unravel_index(idx, latitude.shape)
        
        # judge if in situ in the image
        if not (0 <= pixel_row < latitude.shape[0] and 0 <= pixel_col < latitude.shape[1]):
            logging.warning(f'Point ({lat}, {lon}) is outside image range')
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            continue
            
        logging.info(f'Point ({lat}, {lon}) matched at pixel ({pixel_row}, {pixel_col})')
        
        # create 3*3 window
        window = ((max(0, pixel_row - 1), min(latitude.shape[0], pixel_row + 2)),
                 (max(0, pixel_col - 1), min(latitude.shape[1], pixel_col + 2)))
        
        window_size = (window[0][1] - window[0][0]) * (window[1][1] - window[1][0])
        if window_size < 9:
            logging.warning(f'Incomplete window size: {window_size} pixels')
            results.append(np.concatenate([point_information, np.array([np.nan] * len(bands))]))
            continue
        
        try:
            # extract matchups with quality control
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
                
            cv = np.std(valid_pixels) / mean_value #coefficient of variation
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
            # ensure 3*3 window within the image
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
                print('Fail: No valid pixels !')
            else:                
                valid_ratio = valid_pixels.size / data_box.size
                
            if not valid_ratio > 0.5:
                logging.warning(f'Valid pixel ratio ({valid_ratio:.2f}) < 0.5 for point ({lat}, {lon})')
                results.append(None)
                print('Fail: Valid pixels < 0.5 !')
            else:                
                mean_value = np.mean(valid_pixels)
                
            if mean_value == 0:
                logging.warning(f'Mean value is zero for point ({lat}, {lon})')
                results.append(None)
                print('Fail: Mean value = 0')
            else:                
                cv = np.std(valid_pixels) / mean_value
            if not cv < 0.15:
                logging.warning(f'CV ({cv:.3f}) > 0.15 for point ({lat}, {lon})')
                results.append(None)
                print('Fail: CV > 0.15')
            else:
                print('Successful matched!')
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
    """group data"""
    df = pd.read_csv(path_df)
    results = []
    
    # grouped data by property
    df_sorted = df.sort_values('Satellite_Name')
    grouped_data = df_sorted.groupby('Satellite_Name')
    
    if ac_method == "acolite": #acolite, AR, ocsmart, seadas
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
                        
                        batch_results = geomatching_nc(latitude, longitude, image_data,  #geomatching_tif for tif format
                                                  group_df, bands)
                        results.extend(batch_results)
                else:
                    logging.warning(f"No file found for {satellite_name}")
                    # return empty if didn't find the corresponding file
                    for _, point in group_df.iterrows():
                        results.append(np.concatenate([
                            np.array(point.tolist()),
                            np.array([np.nan] * len(bands))
                        ]))
                    
            except Exception as e:
                logging.error(f"Error processing {satellite_name}: {e}")
                # return empty if no matching
                for _, point in group_df.iterrows():
                    results.append(np.concatenate([
                        np.array([point['Lat'], point['Lon'], point['Point'], point['Satellite_Name']]),
                        np.array([np.nan] * len(bands))
                    ]))
                continue
                
    elif ac_method == "ocsmart":
        for satellite_name, group_df in grouped_data:
            if 'LC08' in satellite_name:
                bands = ['Rrs_443nm', 'Rrs_482nm', 'Rrs_561nm', 'Rrs_655nm']
            # correct band settings in the future application
            elif 'LC09' in satellite_name:
                bands = ['Rrs_443nm', 'Rrs_482nm', 'Rrs_561nm', 'Rrs_655nm']

            elif 'S2A' in satellite_name:
                bands = ['Rrs_443nm', 'Rrs_482nm', 'Rrs_561nm', 'Rrs_655nm']

            elif 'S2B' in satellite_name:
                bands = ['Rrs_443nm', 'Rrs_482nm', 'Rrs_561nm', 'Rrs_655nm']

            else:
                print("Sensor cannot be identified!")
                continue
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
                    # return empty if didn't find the corresponding file
                    for _, point in group_df.iterrows():
                        results.append(np.concatenate([
                            np.array([point['Latitude'], point['Longitude']]),
                            np.array([np.nan] * len(bands))
                        ]))
                    
            except Exception as e:
                logging.error(f"Error processing {satellite_name}: {e}")
                # return empty if no matching
                for _, point in group_df.iterrows():
                    results.append(np.concatenate([
                        np.array([point['Latitude'], point['Longitude'], point['GLORIA_ID'], point['Satellite_Name']]),
                        np.array([np.nan] * len(bands))
                    ]))
                continue
    
    elif ac_method == "AR":
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
    elif ac_method == "seadas":
        bands = ['Rrs_443', 'Rrs_482', 'Rrs_561', 'Rrs_655', 'Rrs_865']

        for satellite_name, group_df in grouped_data:
            try:
                _file = os.path.join(path_image, f"{satellite_name}.L2_OC.nc")

                if os.path.exists(_file):
                    logging.info(f"\nProcessing {satellite_name} with {len(group_df)} points...")
                    with h5py.File(_file, 'r') as ds:
                        latitude = np.array(ds['navigation_data/latitude'])
                        longitude = np.array(ds['navigation_data/longitude'])
                        image_data = np.array([np.array(ds[f'geophysical_data/{band}']*ds[f'geophysical_data/{band}'].attrs['scale_factor']
                                                        +ds[f'geophysical_data/{band}'].attrs['add_offset']) for band in bands])

                        batch_results = geomatching_nc(latitude, longitude, image_data,
                                                       group_df, bands)
                        results.extend(batch_results)
                else:
                    logging.warning(f"No file found for {satellite_name}")
                    # return empty if didn't find the corresponding file
                    for _, point in group_df.iterrows():
                        results.append(np.concatenate([
                            np.array([point['Latitude'], point['Longitude']]),
                            np.array([np.nan] * len(bands))
                        ]))

            except Exception as e:
                logging.error(f"Error processing {satellite_name}: {e}")
                # return empty if no matching
                for _, point in group_df.iterrows():
                    results.append(np.concatenate([
                        np.array([point['Latitude'], point['Longitude'], point['GLORIA_ID'], point['Satellite_Name']]),
                        np.array([np.nan] * len(bands))
                    ]))
                continue
    # create DataFrame for results
    if results:
        columns = df.columns.tolist() + bands
        if ac_method == "AR":
            df_result = pd.DataFrame(results)
        else:
            df_result = pd.DataFrame(results, columns=columns)
        return df_result
    else:
        logging.warning("No results found")
        # return empty DataFrame
        return pd.DataFrame(columns=df.columns.tolist() + bands)

def main():
    """main function：matching and save"""
    # Parameters
    config = {
        'ac_method': "acolite",  # Options: "acolite" or "ocsmart" or "seadas" or "AR"
        'path_df': r'D:\10-Landsat_Aquatic_Reflectance\LakeLanier_matchups\LakeLanier_20241008_matchups_landsat8.csv', #in situ table
        'path_image': r'D:\10-Landsat_Aquatic_Reflectance\LakeLanier_matchups\L2\LC08_L2TP_019036_20241008_20241008_02_RT',
        'output_dir': r'D:\10-Landsat_Aquatic_Reflectance\LakeLanier_matchups\results-2'
    }
    
    # create output dir
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # log file setting
    logger, file_handler = setup_logging(config['output_dir'])
    
    try:
        # start time
        start_time = datetime.now()
        logger.info(f"Processing started at {start_time}")
        logger.info(f"Using atmospheric correction method: {config['ac_method']}")
        
        # processing data
        results_df = process_data(config['path_df'], config['path_image'], config['ac_method'])
        
        if results_df is not None and not results_df.empty:
            # generate output filename
            output_filename = f"matchups_{config['ac_method']}_{start_time.strftime('%Y%m%dT%H%M%S')}.csv"
            output_path = os.path.join(config['output_dir'], output_filename)
            
            # save matching results
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Total matched points: {len(results_df)}")
            
            # generate matching information
            stats = {
                'total_input_points': len(pd.read_csv(config['path_df'])),
                'successful_matches': len(results_df),
                'match_rate': len(results_df) / len(pd.read_csv(config['path_df'])) * 100
            }
            
            logger.info("Matching Statistics:")
            logger.info(f"Total input points: {stats['total_input_points']}")
            logger.info(f"Successful matches: {stats['successful_matches']}")
            logger.info(f"Match rate: {stats['match_rate']:.2f}%")
            
            # save statics
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
        # start time and end time recording
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Processing completed at {end_time}")
        logger.info(f"Total processing time: {duration}")
        
        # close file
        logger.removeHandler(file_handler)
        file_handler.close()

if __name__ == '__main__':
    # warning
    warnings.filterwarnings('always')  
    
    try:
        main()
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error occurred in main program: {str(e)}")
        raise