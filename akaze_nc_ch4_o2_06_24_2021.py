#   Author: Amir Souri (ahsouri@gmail.com)
#   Date: Dec 5, 2020

# importing libraries
import cv2
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import sys
from skimage.measure import LineModelND, ransac
from scipy import stats
import glob
from scipy.interpolate import  interpolate, griddata
import rasterio
import utm
import scipy.io
import os.path
from os import path
# main function
def main(o2_files,ch4_files):
    # specify inputs:
    grid_size = 0.0001 # define the grid size of the mosaic in degree (0.01 deg~1.1km)  
    do_histeq = True  # do histogram eq to enhance the contrast
    # specify O2 folder/files
    #o2_file_bundle = 'MethaneAIR_L1B_O2_20191112T2119'
    #output_file = 'output_akaze_' + o2_file_bundle +'_'+ ch4_file_bundle + '.nc'
    # read o2 channel, lats and lons
    methaneair_o2,methaneair_lon,methaneair_lat = read_methaneair_o2(o2_files,1,w1=0,w2=167)
    # mosaicing
    mosaic_o2,lats_o2,lons_o2=mosaic(methaneair_o2,methaneair_lon,methaneair_lat,grid_size)
    #mosaic_o2 = np.transpose(mosaic_o2)
    #lons_o2 = np.transpose(lons_o2)
    #lats_o2 = np.transpose(lats_o2)
    # read ch4 channel, lats and lons
    methaneair_ch4,methaneair_lon,methaneair_lat = read_methaneair(ch4_files,1,w1=273,w2=370)
    print(len(methaneair_lon))
    # mosaicing
    mosaic_ch4,lats_ch4,lons_ch4=mosaic(methaneair_ch4,methaneair_lon,methaneair_lat,grid_size)
    # histeq
    if do_histeq == True:
       clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(10,10))
       mosaic_o2_enh = clahe.apply(np.uint8(mosaic_o2*255))
       mosaic_ch4_enh = clahe.apply(np.uint8(mosaic_ch4*255))
    else: # akaze requires uint8 type anyway
       mosaic_o2_enh = np.uint8(mosaic_o2*255)
       mosaic_ch4_enh = np.uint8(mosaic_ch4*255)
       # akaze
    slope_lat,slope_lon,intercept_lat,intercept_lon,r_lat,r_lon=akaze(mosaic_ch4_enh,mosaic_o2_enh,10000,lats_o2,lons_o2,lats_ch4,lons_ch4)
    # save the output
    print(r_lat)
    print(r_lon)
    filename = str(ch4_files[0]).split(".nc")[0]
    filename=filename.split("/")[-1]
    filename = str(filename)+'_relative_correction_akaze.txt'
    if path.isfile(filename):
       file1 = open(filename, "a")
       L =  str(slope_lon) +',' + str(slope_lat) + ',' +str(intercept_lon) +','+ str(intercept_lat) + ',' + str(r_lon) +',' + str(r_lat)
       file1.writelines(L)
    else:
       L2 = str(slope_lon) +',' + str(slope_lat) + ',' +str(intercept_lon) +','+ str(intercept_lat) + ',' + str(r_lon) +',' + str(r_lat)
       file1 = open(filename, "w")
       file1.writelines(L2)
    return(slope_lat,slope_lon,intercept_lat,intercept_lon,r_lat,r_lon)
# reading netcdf files
def read_netcdf(filename,var):
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    var = nc_fid.variables[var]
    return np.squeeze(var)
def read_group_nc(filename,num_groups,group,var):
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    if num_groups == 1:
       out = np.array(nc_fid.groups[group].variables[var])
    elif num_groups == 2:
       out = np.array(nc_fid.groups[group[0]].groups[group[1]].variables[var])
    elif num_groups == 3:
       out = np.array(nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].variables[var])
    nc_fid.close()
    return np.squeeze(out)
# methaneair reader function    
def read_methaneair_o2(files,bd,w1,w2):
    methaneair_rad =  []
    methaneair_lat = []
    methaneair_lon = []
    for i in range(len(files)):
        file = files[i]+'.nc'
        rad_o2 = read_group_nc(file,1,'Band' + str(bd),'Radiance')
        lat_o2 = read_group_nc(file,1,'Geolocation','Latitude')
        lon_o2 = read_group_nc(file,1,'Geolocation','Longitude')
        rad_o2 [rad_o2 <= 0] = np.nan
        lon_o2[:,:] = lon_o2[:,::-1]
        lat_o2[:,:] = lat_o2[:,::-1]
        rad_o2 = np.nanmean(rad_o2[w1:w2,:,::-1],axis=0)
        methaneair_rad.append(rad_o2)
        methaneair_lon.append(lon_o2)
        methaneair_lat.append(lat_o2)
    return methaneair_rad,methaneair_lon,methaneair_lat
def read_methaneair(files,bd,w1,w2):
    methaneair_rad =  []
    methaneair_lat = []
    methaneair_lon = []
    for i in range(len(files)):
        file = files[i]+'.nc'
        rad_o2 = read_group_nc(file,1,'Band' + str(bd),'Radiance')
        lat_o2 = read_group_nc(file,1,'Geolocation','Latitude')
        lon_o2 = read_group_nc(file,1,'Geolocation','Longitude')
        rad_o2 [rad_o2 <= 0] = np.nan
        lon_o2[:,:] = lon_o2[:,::-1]
        lat_o2[:,:] = lat_o2[:,::-1]
        rad_o2 = np.nanmean(rad_o2[w1:w2,:,::-1],axis=0)
        methaneair_rad.append(rad_o2)
        methaneair_lon.append(lon_o2)
        methaneair_lat.append(lat_o2)
    return methaneair_rad,methaneair_lon,methaneair_lat
# putting tiles together in a defined grid
def mosaic(methaneair,methaneair_lon,methaneair_lat,grid_size):
    # first finding corners
    max_lat = []
    min_lat = []
    max_lon = []
    min_lon = []
    for i in range(len(methaneair)):
        min_lat.append(np.nanmin(methaneair_lat[i]))
        max_lat.append(np.nanmax(methaneair_lat[i]))
        min_lon.append(np.nanmin(methaneair_lon[i]))
        max_lon.append(np.nanmax(methaneair_lon[i]))
    min_lat=np.nanmin(min_lat)
    max_lat=np.nanmax(max_lat)
    min_lon=np.nanmin(min_lon)
    max_lon=np.nanmax(max_lon)
    lon = np.arange(min_lon,max_lon,grid_size)
    lat = np.arange(min_lat,max_lat,grid_size)  
    lons,lats = np.meshgrid(lon,lat)
    full_mosaic = np.zeros((np.shape(lons)[0],np.shape(lons)[1],len(methaneair)))
    for i in range(len(methaneair)):
        lon_metair = methaneair_lon[i]
        lat_metair = methaneair_lat[i]
        metair = methaneair[i]  
        points = np.zeros((np.size(lon_metair),2))
        points[:,0] = lon_metair.flatten()
        points[:,1] = lat_metair.flatten()
        full_mosaic[:,:,i] = griddata(points, metair.flatten(), (lons, lats), method='linear')
    mosaic_air = np.zeros_like(lats)    
    for i in range(np.shape(full_mosaic)[0]):
        for j in range(np.shape(full_mosaic)[1]):
            temp = []
            for k in range(len(methaneair)):
                temp.append(full_mosaic[i,j,k])
            temp = np.array(temp)
            temp[temp==0]=np.nan
            mosaic_air[i,j] = np.nanmean(temp)
    mosaic_air = cv2.normalize(mosaic_air,np.zeros(mosaic_air.shape, np.double),1.0,0.0,cv2.NORM_MINMAX)
    #mosaic_o2 = cv2.equalizeHist(np.uint8(mosaic_o2*255))
    #plt.imshow(mosaic_o2)
    #plt.show()
    return mosaic_air,lats,lons
def akaze(master,slave,dist_thr,lat_master,lon_master,lat_slave,lon_slave):

    #keypoints
    akaze_mod = cv2.AKAZE_create()
    keypoints_1, descriptors_1 = akaze_mod.detectAndCompute(master,None)
    keypoints_2, descriptors_2 = akaze_mod.detectAndCompute(slave,None)
    bf = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    #DMatch.distance - Distance between descriptors. The lower, the better it is.
    #DMatch.trainIdx - Index of the descriptor in train descriptors
    #DMatch.queryIdx - Index of the descriptor in query descriptors
    #DMatch.imgIdx - Index of the train image.
    master_matched,slave_matched=find_matched_i_j(matches,keypoints_1,keypoints_2,dist_thr)
    lat_1 =[]
    lon_1 =[]
    lat_2 =[]
    lon_2 =[]
    for i in range(np.shape(master_matched)[0]):
        lat_2.append(lat_master[int(np.round(master_matched[i,1])),int(np.round(master_matched[i,0]))])
        lon_2.append(lon_master[int(np.round(master_matched[i,1])),int(np.round(master_matched[i,0]))])
        lat_1.append(lat_slave[int(np.round(slave_matched[i,1])),int(np.round(slave_matched[i,0]))])
        lon_1.append(lon_slave[int(np.round(slave_matched[i,1])),int(np.round(slave_matched[i,0]))])

    lat_1=np.array(lat_1)
    lat_2=np.array(lat_2)
    lon_1=np.array(lon_1)
    lon_2=np.array(lon_2)

    pts1=np.zeros((len(master_matched),2))
    pts2=np.zeros((len(master_matched),2))

    pts1[:,0] = lon_1
    pts1[:,1] = lat_1
    pts2[:,0] = lon_2
    pts2[:,1] = lat_2
    #img_1 = cv2.drawKeypoints(master,keypoints_1,master)
    #plt.imshow(img_1)
    #plt.show()
    #img_1 = cv2.drawKeypoints(slave,keypoints_2,slave)
    #plt.imshow(img_1)
    #plt.show()
    
    print('number of matched points: ' + str(len(matches)))
    
    data = np.column_stack([lat_1, lat_2])
    good_lat1, good_lat2 = robust_inliner(data,False)
    slope_lat, intercept_lat, r_value1, p_value, std_err = stats.linregress(good_lat1,good_lat2)
    #print(r_value)
    data = np.column_stack([lon_1, lon_2])
    good_lon1, good_lon2 = robust_inliner(data,False)
    slope_lon, intercept_lon, r_value2, p_value, std_err = stats.linregress(good_lon1,good_lon2)
    #print(r_value)
    return slope_lat,slope_lon,intercept_lat,intercept_lon,r_value1,r_value2
def find_matched_i_j(matches_var,keypoints1,keypoints2,dist_thr):
    # Initialize lists
    list_kp1 = []
    list_kp2 = []
    # For each match...
    for mat in matches_var:
    # Get the matching keypoints for each of the images
        if mat.distance>dist_thr:
            continue
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
         # x - columns
         # y - rows
         # Get the coordinates
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt
         # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
    list_kp1 = np.array(list_kp1)
    list_kp2 = np.array(list_kp2)
    return list_kp1,list_kp2
def robust_inliner(data,doplot):
# Fit line using all data

    model = LineModelND()
    model.estimate(data)
    
    # Robustly fit linear model with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                               residual_threshold=0.0005, max_trials=100000)
    outliers = inliers == False
    # Predict data of estimated models
    line_x = np.arange(-360, 360)
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_x)
    # Compare estimated coefficients
    if doplot == True:
       fig, ax = plt.subplots()
       ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
           label='Inlier data')
       ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
           label='Outlier data')
       ax.plot(line_x, line_y, '-k', label='Line model from all data')
       ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
       ax.legend(loc='lower left')
       plt.xlim(np.min(data[:,0])-0.01,np.max(data[:,0])+0.01)
       plt.ylim(np.min(data[:,1])-0.01,np.max(data[:,1])+0.01)
       plt.show()
    return data[inliers, 0],data[inliers, 1]
def write_to_nc(output_file,mosaic_ch4,mosaic_o2,lats_o2,lons_o2,lats_ch4,lons_ch4,lats_ch4_new,lons_ch4_new):
    ncfile = Dataset(output_file,'w') 
    # create the x and y dimensions.
    ncfile.createDimension('x',np.shape(mosaic_ch4)[0])
    ncfile.createDimension('y',np.shape(mosaic_ch4)[1])
    ncfile.createDimension('xx',np.shape(mosaic_o2)[0])
    ncfile.createDimension('yy',np.shape(mosaic_o2)[1])
    data1 = ncfile.createVariable('mosaic_ch4',dtype('uint8').char,('x','y'))
    data1[:,:] = mosaic_ch4
    data2 = ncfile.createVariable('mosaic_o2',dtype('uint8').char,('xx','yy'))
    data2[:,:] = mosaic_o2
    data3 = ncfile.createVariable('lats_o2',dtype('float64').char,('xx','yy'))
    data3[:,:] = lats_o2
    data4 = ncfile.createVariable('lons_o2',dtype('float64').char,('xx','yy'))
    data4[:,:] = lons_o2
    data5 = ncfile.createVariable('lats_ch4_new',dtype('float64').char,('x','y'))
    data5[:,:] = lats_ch4_new
    data6 = ncfile.createVariable('lons_ch4_new',dtype('float64').char,('x','y'))
    data6[:,:] = lons_ch4_new
    data5 = ncfile.createVariable('lats_ch4',dtype('float64').char,('x','y'))
    data5[:,:] = lats_ch4
    data6 = ncfile.createVariable('lons_ch4',dtype('float64').char,('x','y'))
    data6[:,:] = lons_ch4
    ncfile.close()

if __name__ == '__main__':
    main()
