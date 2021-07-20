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
import scipy.io
import os.path
from os import path
import os
# main function
def main(o2_files,computer):
    # specify inputs:
    if (computer == 'Odyssey'):
        file_landsat = '/n/holylfs04/LABS/wofsy_lab/Lab/econway/DATA/landsat_set2.nc'
    elif (computer == 'Hydra'):
        file_landsat = '/home/econway/DATA/landsat_set2.nc'
    grid_size = 0.0001 # define the grid size of the mosaic in degree (0.01 deg~1.1km)  
    do_histeq = True  # do histogram eq to enhance the contrast
    # specify O2 folder/files
    #o2_file_bundle = str(argv)
    filesavename = os.path.basename(o2_files[0])
    #filesavename = str(filesavename).split('.nc')[0] 
    output_file = 'output_akaze_'+str(filesavename)+'.txt'
    #o2_files = os.path.join(o2_folder+'/'+str(o2_file_bundle)+'.nc')
    # read o2 channel, lats and lons
    methaneair_o2,methaneair_lon,methaneair_lat = read_methaneair(o2_files)
    # mosaicing
    moasic_o2,lats_o2,lons_o2=mosaic(methaneair_o2,methaneair_lon,methaneair_lat,grid_size)
    # landsat file and reader (cropped to the moasic)
    landsat_gray = landsat_read(file_landsat,lats_o2,lons_o2)
    # histeq
    if do_histeq == True:
       #moasic_o2 = cv2.equalizeHist(np.uint8(moasic_o2*255))
       #landsat_gray = cv2.equalizeHist(np.uint8(landsat_gray*255))
       clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(10,10))
       moasic_o2_enh = clahe.apply(np.uint8(moasic_o2*255))
       landsat_gray_enh = clahe.apply(np.uint8(landsat_gray*255))
    else: # akaze requires uint8 type anyway
       moasic_o2 = np.uint8(moasic_o2*255)
       landsat_gray = np.uint8(landsat_gray*255)
       # akaze
    slope_lat,slope_lon,intercept_lat,intercept_lon,r_lat,r_lon=akaze(landsat_gray_enh,moasic_o2_enh,1000,lats_o2,lons_o2)
    # one more time in case slopes are too large with narrowed scene
    if abs(slope_lat)<0.8 or abs(slope_lat)>1.2 or abs(slope_lon)<0.8 or abs(slope_lon)>1.2:
#########################################################################################
       landsat_gray[moasic_o2==0.0]= 0.0
       landsat_gray[np.isnan(moasic_o2)]= 0.0
       moasic_o2[landsat_gray==0.0]= 0.0
       moasic_o2[np.isnan(landsat_gray)]= 0.0
       clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(10,10))
       moasic_o2_enh = clahe.apply(np.uint8(moasic_o2*255))
       landsat_gray_enh = clahe.apply(np.uint8(landsat_gray*255)) 
       slope_lat,slope_lon,intercept_lat,intercept_lon,r_lat,r_lon=akaze(landsat_gray_enh,moasic_o2_enh,10000,lats_o2,lons_o2)
    #outputtting in txt
    #filename = str(o2_file_bundle)+'_absolute_correction_akaze.txt'
    filename = str(filesavename)+'_absolute_correction_akaze.txt'
    if path.isfile(filename):
       file1 = open(filename, "a")
       L =  str(slope_lon) +',' + str(slope_lat) + ',' +str(intercept_lon) +','+ str(intercept_lat) + \
               ',' + str(r_lon) +',' + str(r_lat)
       file1.writelines(L)
    else:
       L1 = 'file_bundle,slope_lon,slope_lat,intercept_lon,intercept_lat,rvalue_lon,rvalue_lat'
       L2 =  str(slope_lon) +',' + str(slope_lat) + ',' +str(intercept_lon) +','+ str(intercept_lat) + \
               ',' + str(r_lon) +',' + str(r_lat)
       file1 = open(filename, "w")
       file1.writelines(L2)
    #outputting in netcdf
    #    write_to_nc(output_file,landsat_gray_enh,moasic_o2_enh,lats_o2,lons_o2,(lats_o2-intercept_lat)/ \
    #        slope_lat,(lons_o2-intercept_lon)/slope_lon)
    return(slope_lat,slope_lon,intercept_lat,intercept_lon,r_lat,r_lon)
# reading netcdf files
def read_netcdf(filename,var):
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    var = nc_fid.variables[var][:]
    nc_fid.close()
    return np.squeeze(var)
def read_group_nc(filename,num_groups,group,var):
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    if num_groups == 1:
       out = np.array(nc_fid.groups[group].variables[var][:])
    elif num_groups == 2:
       out = np.array(nc_fid.groups[group[0]].groups[group[1]].variables[var][:])
    elif num_groups == 3:
       out = np.array(nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].variables[var][:])
    nc_fid.close()
    return np.squeeze(out)
# methaneair reader function    
def read_methaneair(o2_files):
    methaneair_o2 =  []
    methaneair_lat = []
    methaneair_lon = []
    for i in range(len(o2_files)):
        file = o2_files[i]+'.nc'
        print(file)
        rad_o2 = read_group_nc(file,1,'Band1','Radiance')
        lat_o2 = read_group_nc(file,1,'Geolocation','Latitude')
        lon_o2 = read_group_nc(file,1,'Geolocation','Longitude')
        rad_o2 [rad_o2 <= 0] = np.nan
        lon_o2[:,:] = lon_o2[:,::-1]
        lat_o2[:,:] = lat_o2[:,::-1]
        #rad_o2 = np.nanmean(rad_o2[0:167,:,:],axis=0)
        rad_o2 = np.nanmean(rad_o2[100:450,:,::-1],axis=0)
        methaneair_o2.append(rad_o2)
        methaneair_lon.append(lon_o2)
        methaneair_lat.append(lat_o2)
    return methaneair_o2,methaneair_lon,methaneair_lat
# putting tiles together in a defined grid
def mosaic(methaneair_o2,methaneair_lon,methaneair_lat,grid_size):
    # first finding corners
    max_lat = []
    min_lat = []
    max_lon = []
    min_lon = []
    for i in range(len(methaneair_o2)):
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
    full_moasic = np.zeros((np.shape(lons)[0],np.shape(lons)[1],len(methaneair_o2)))
    for i in range(len(methaneair_o2)):
        lon_metair = methaneair_lon[i]
        lat_metair = methaneair_lat[i]
        metair = methaneair_o2[i]  
        points = np.zeros((np.size(lon_metair),2))
        points[:,0] = lon_metair.flatten()
        points[:,1] = lat_metair.flatten()
        full_moasic[:,:,i] = griddata(points, metair.flatten(), (lons, lats), method='linear')
    moasic_o2 = np.zeros_like(lats)    
    for i in range(np.shape(full_moasic)[0]):
        for j in range(np.shape(full_moasic)[1]):
            temp = []
            for k in range(len(methaneair_o2)):
                temp.append(full_moasic[i,j,k])
            temp = np.array(temp)
            temp[temp==0]=np.nan
            moasic_o2[i,j] = np.nanmean(temp)
    moasic_o2 = cv2.normalize(moasic_o2,np.zeros(moasic_o2.shape, np.double),1.0,0.0,cv2.NORM_MINMAX)
    #moasic_o2 = cv2.equalizeHist(np.uint8(moasic_o2*255))
    #plt.imshow(moasic_o2)
    #plt.show()
    return moasic_o2,lats,lons
def landsat_read(file_landsat,lats_o2,lons_o2):
    lat_landsat = read_netcdf(file_landsat,'lat_landsat')
    lon_landsat = read_netcdf(file_landsat,'lon_landsat')
    landsat_gray = read_netcdf(file_landsat,'landsat_gray')
    # resizing landsat
    #small_landsat = cv2.resize(landsat_gray, (0,0), fx=0.3, fy=0.3)
    #small_lat = cv2.resize(lat_landsat, (0,0), fx=0.3, fy=0.3)
    #small_lon =	cv2.resize(lon_landsat, (0,0), fx=0.3, fy=0.3) 
    #landsat_gray = small_landsat
    #lat_landsat = small_lat
    #lon_landsat = small_lon
    lon_range = np.array([min(lons_o2.flatten()),max(lons_o2.flatten())])
    lat_range = np.array([min(lats_o2.flatten()),max(lats_o2.flatten())])
    cost = np.sqrt((lon_landsat-lon_range[0])**2+(lat_landsat-lat_range[0])**2)
    indices_xy_1 = np.array(np.argwhere(cost==np.min(np.min(cost))))
    cost = np.sqrt((lon_landsat-lon_range[1])**2+(lat_landsat-lat_range[1])**2)
    indices_xy_2 = np.array(np.argwhere(cost==np.min(np.min(cost))))
    lon_landsat=np.transpose(lon_landsat[indices_xy_1[0,0]:indices_xy_2[0,0],indices_xy_2[0,1]:indices_xy_1[0,1]])
    lat_landsat=np.transpose(lat_landsat[indices_xy_1[0,0]:indices_xy_2[0,0],indices_xy_2[0,1]:indices_xy_1[0,1]])
    landsat_gray=landsat_gray[indices_xy_2[0,1]:indices_xy_1[0,1],indices_xy_1[0,0]:indices_xy_2[0,0]]
    #subset landsat to methaneair
    points = np.zeros((np.size(lat_landsat),2))
    points[:,0] = lon_landsat.flatten()
    points[:,1] = lat_landsat.flatten()
    landsat_gray = griddata(points, landsat_gray.flatten(), (lons_o2, lats_o2), method='linear')
    landsat_gray = cv2.normalize(landsat_gray,np.zeros(landsat_gray.shape, np.double),1.0,0.0,cv2.NORM_MINMAX)
    #moasic_o2 = cv2.normalize(moasic_o2,np.zeros(moasic_o2.shape, np.double),1.0,0.0,cv2.NORM_MINMAX)
    #landsat_gray = cv2.equalizeHist(np.uint8(landsat_gray*255))
    #plt.imshow(landsat_gray)
    #plt.show()
    return landsat_gray
def akaze(master,slave,dist_thr,lat_master,lon_master):

    #keypoints
    akaze_mod = cv2.AKAZE_create()
    keypoints_1, descriptors_1 = akaze_mod.detectAndCompute(master,None)
    keypoints_2, descriptors_2 = akaze_mod.detectAndCompute(slave,None)
    bf = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING, crossCheck=True)
    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    #nn_matches = matcher.knnMatch(descriptors_1, descriptors_2, 2)
    #matched1 = []
    #matched2 = []
    #nn_match_ratio = 0.8 # Nearest neighbor matching ratio
    #for m, n in nn_matches:
    #    if m.distance < nn_match_ratio * n.distance:
    #       matched1.append(keypoints_1[m.queryIdx])
    #       matched2.append(keypoints_2[m.trainIdx])
    #matches = sorted(matches, key = lambda x:x.distance)
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
        lat_1.append(lat_master[int(np.round(master_matched[i,1])),int(np.round(master_matched[i,0]))])
        lon_1.append(lon_master[int(np.round(master_matched[i,1])),int(np.round(master_matched[i,0]))])
        lat_2.append(lat_master[int(np.round(slave_matched[i,1])),int(np.round(slave_matched[i,0]))])
        lon_2.append(lon_master[int(np.round(slave_matched[i,1])),int(np.round(slave_matched[i,0]))])

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
    
    #fig = plt.figure()
    #img_1 = cv2.drawKeypoints(master,keypoints_1,master)
    #plt.imshow(img_1)
    #plt.show()
    #fig.savefig('plot_master.png',dpi=1000)
    #plt.close(fig)

    #fig = plt.figure()
    #img_1 = cv2.drawKeypoints(slave,keypoints_2,slave)
    #plt.imshow(img_1)
    #fig.savefig('plot_slave.png',dpi=1000)
    #plt.close(fig)
    #plt.show()
    
    print('number of matched points: ' + str(len(master_matched)))
    
    data = np.column_stack([lat_1, lat_2])
    good_lat1, good_lat2 = robust_inliner(data,False,'./lat_pic')
    slope_lat, intercept_lat, r_value1, p_value, std_err = stats.linregress(good_lat1,good_lat2)
    #print(r_value1)
    data = np.column_stack([lon_1, lon_2])
    good_lon1, good_lon2 = robust_inliner(data,False,'./lon_pic')
    slope_lon, intercept_lon, r_value2, p_value, std_err = stats.linregress(good_lon1,good_lon2)
    #print(r_value2)
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
def robust_inliner(data,doplot,file_plot):
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
       #plt.show()
       fig.savefig(file_plot + '.png',dpi=300)
       plt.close(fig)
    return data[inliers, 0],data[inliers, 1]
def write_to_nc(output_file,landsat_gray,moasic_o2,lats_o2_old,lons_o2_old,lats_o2_new,lons_o2_new):
    ncfile = Dataset(output_file,'w') 
    # create the x and y dimensions.
    ncfile.createDimension('x',np.shape(landsat_gray)[0])
    ncfile.createDimension('y',np.shape(landsat_gray)[1])
    data1 = ncfile.createVariable('landsat_gray',dtype('uint8').char,('x','y'))
    data1[:,:] = landsat_gray
    data2 = ncfile.createVariable('moasic_o2',dtype('uint8').char,('x','y'))
    data2[:,:] = moasic_o2
    data3 = ncfile.createVariable('lats_o2_old',dtype('float64').char,('x','y'))
    data3[:,:] = lats_o2_old    
    data4 = ncfile.createVariable('lons_o2_old',dtype('float64').char,('x','y'))
    data4[:,:] = lons_o2_old   
    data5 = ncfile.createVariable('lats_o2_new',dtype('float64').char,('x','y'))
    data5[:,:] = lats_o2_new       
    data6 = ncfile.createVariable('lons_o2_new',dtype('float64').char,('x','y'))
    data6[:,:] = lons_o2_new  
    ncfile.close()      

