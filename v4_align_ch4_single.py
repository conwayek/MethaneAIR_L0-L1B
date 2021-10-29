import datetime
from filelock import FileLock
import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate,optimize
from lmfit import minimize, Parameters, Parameter, printfuncs, fit_report
from scipy import signal
import cv2
from netCDF4 import Dataset
import netCDF4 as nc4
import time
import os
import pysplat
from skimage.measure import block_reduce




class Alignment(object):

    def __init__(self,inputfile,dest):
        self.t0 = time.time()
        self.dest = dest
        self.logfile_ch4_native = os.path.join(self.dest,'CH4_NATIVE/log_file.txt') 
        g = open(self.logfile_ch4_native,'r')
        native = g.readlines()  
        g.close()
        nfiles = len(native)
         
        self.native_ch4_files = []
        self.native_ch4_priority = []
        
        for i in range(nfiles):
            self.native_ch4_files.append(os.path.basename(native[i].split(' ')[0]))
            self.native_ch4_priority.append((native[i].split(' ')[1]).split('\n')[0])


        ### GET THE LIST OF CH4 FILES FIRST
        ch4dir = os.path.join(self.dest,'CH4_NATIVE/')
        self.CH4Files = []
        self.CH4Times = []
        self.CH4StartTime = []
        self.CH4EndTime = []
        count=0
        for file in os.listdir(ch4dir):
            if file.endswith(".nc"):
                count=count+1
                base=os.path.basename(file)
                name=(base.split(".nc"))[0]
                start = name[28:34]
                end = name[44:50]
                name = name[19:66]
                self.CH4Files.append(os.path.join(ch4dir, file))
                self.CH4Times.append(name)
                self.CH4EndTime.append(end)
                self.CH4StartTime.append(start)
    
    
        ### GET THE LIST OF O2 FILES NEXT
        o2dir = os.path.join(self.dest,'O2_NATIVE/')
        self.O2Files = []
        self.O2Times = []
        self.O2StartTime = []
        self.O2EndTime = []
        for file in os.listdir(o2dir):
            count=count+1
            if file.endswith(".nc"):
                base=os.path.basename(file)
                name=(base.split(".nc"))[0]
                start = name[27:33]
                end = name[43:49]
                name = name[18:65]
                self.O2Files.append(os.path.join(o2dir, file))
                self.O2Times.append(name)
                self.O2EndTime.append(end)
                self.O2StartTime.append(start)

        self.inputCH4Times = []
        self.inputCH4StartTime = []
        self.inputCH4EndTime = []
        base=os.path.basename(inputfile)
        name=(base.split(".nc"))[0]
        start = name[28:34]
        end = name[44:50]
        name = name[19:66]
        self.inputCH4Times=name
        self.inputCH4EndTime=end
        self.inputCH4StartTime=start
        
        self.hseconds = abs(np.int(self.inputCH4EndTime)) % 100
        self.hminutes = ((abs(np.int(self.inputCH4EndTime)) % 10000) - self.hseconds)/100
        self.hhour = (np.int(self.inputCH4EndTime) - self.hseconds - self.hminutes*100 ) / 10000 
        self.delta = np.int(self.inputCH4EndTime) - np.int(self.inputCH4StartTime)

        x = str(os.path.basename(inputfile))
        x = x.split('CH4_')[1]
        x = x.split('_')

        startch4seconds =  datetime.datetime.strptime(x[0],"%Y%m%dT%H%M%S").second
        startch4minutes = datetime.datetime.strptime(x[0],"%Y%m%dT%H%M%S").minute
        startch4hour = datetime.datetime.strptime(x[0],"%Y%m%dT%H%M%S").hour

        endch4seconds =  datetime.datetime.strptime(x[1],"%Y%m%dT%H%M%S").second
        endch4minutes = datetime.datetime.strptime(x[1],"%Y%m%dT%H%M%S").minute
        endch4hour = datetime.datetime.strptime(x[1],"%Y%m%dT%H%M%S").hour
        for k in range(1):
            done = False
            for j in range(len(self.O2Times)):
                starto2seconds =  datetime.datetime.strptime(self.O2StartTime[j],"%H%M%S").second
                starto2minutes = datetime.datetime.strptime(self.O2StartTime[j],"%H%M%S").minute
                starto2hour = datetime.datetime.strptime(self.O2StartTime[j],"%H%M%S").hour

                endo2seconds =  datetime.datetime.strptime(self.O2EndTime[j],"%H%M%S").second
                endo2minutes = datetime.datetime.strptime(self.O2EndTime[j],"%H%M%S").minute
                endo2hour = datetime.datetime.strptime(self.O2EndTime[j],"%H%M%S").hour

                #if((self.CH4StartTime == self.O2StartTime[j])   and (  self.CH4EndTime == self.O2EndTime[j]  )   ):
                if((done == False) and (math.isclose(startch4seconds,starto2seconds,abs_tol=1.1)) and (math.isclose(endch4seconds,endo2seconds,abs_tol=1.1)) and 
                     (startch4minutes == starto2minutes) and (startch4hour == starto2hour) 
                    and (endch4minutes == endo2minutes) and (endch4hour == endo2hour) ):
                    done = True
                    #CH4
                    filech4 = inputfile 
                    #O2
                    fileo2 = self.O2Files[j]
                    # Let us align the channels/files 
                    self.align(filech4,fileo2)
        if(done == False):
           self.align(inputfile,None)

    def align(self,filech4,fileo2):
        datach4 = Dataset(filech4)
        self.x1 = datach4.groups['Band1']
        self.x2 = datach4.groups['Geolocation']
        self.x3 = datach4.groups['SupportingData']
        radch4 = self.x1['Radiance'][:,:,:].data

        if(fileo2 is not None):
            datao2 = Dataset(fileo2)
            y1 = datao2.groups['Band1']
            y2 = datao2.groups['Geolocation']
            y3 = datao2.groups['SupportingData']


            akaze_used = y3['AkazeUsed'][:]
            optimized_used = y3['OptimizedUsed'][:]
            avionics_used = y3['AvionicsUsed'][:]

            rado2 = y1['Radiance'][:,:,:].data
                 
        else:
            datao2 = Dataset(filech4)
            y1 = datao2.groups['Band1']
            y2 = datao2.groups['Geolocation']
            y3 = datao2.groups['SupportingData']


            akaze_used = y3['AkazeUsed'][:]
            optimized_used = y3['OptimizedUsed'][:]
            avionics_used = y3['AvionicsUsed'][:]

        wvl_new = np.zeros(y1['Wavelength'].shape )
        rad_err_new = np.zeros(y1['RadianceUncertainty'].shape)
        rad_flags_new = np.zeros(y1['RadianceFlag'].shape)
        rad_new = np.zeros(y1['Radiance'].shape   )
        
        corner_lon_new = np.zeros(y2['CornerLongitude'].shape)
        corner_lat_new = np.zeros(y2['CornerLatitude'].shape)
        lon_new = np.zeros(y2['Longitude'].shape                            )
        lat_new = np.zeros(y2['Latitude'].shape                             )
        sza_new = np.zeros(y2['SolarZenithAngle'].shape        )
        vza_new = np.zeros(y2['ViewingZenithAngle'].shape    )
        vaa_new = np.zeros(y2['ViewingAzimuthAngle'].shape  )
        saa_new = np.zeros(y2['SolarAzimuthAngle'].shape  )
        aza_new = np.zeros(y2['RelativeAzimuthAngle'].shape)
        surfalt_new = np.zeros(y2['SurfaceAltitude'].shape)
        obsalt_new = np.zeros(y2['ObservationAltitude'].shape)
        time_new = np.zeros(y2['Time'].shape)
        ac_lon_new = np.zeros(y2['AircraftLongitude'].shape)
        ac_lat_new = np.zeros(y2['AircraftLatitude'].shape)
        ac_alt_surf_new = np.zeros(y2['AircraftAltitudeAboveSurface'].shape)
        ac_surf_alt_new = np.zeros(y2['AircraftSurfaceAltitude'].shape)
        ac_pix_bore_new = np.zeros(y2['AircraftPixelBore'].shape)
        ac_pos_new = np.zeros(y2['AircraftPos'].shape)

        akaze_used_new = self.x3['AkazeUsed'][0]
        optimized_used_new = self.x3['OptimizedUsed'][0]
        avionics_used_new = self.x3['AvionicsUsed'][0]

        if(akaze_used_new==0 and optimized_used_new==0):
            akaze_msi_new = None

        if(optimized_used_new==1): 
            ac_roll_new = np.zeros(y2['Time'].shape)
            ac_pitch_new = np.zeros(y2['Time'].shape)
            ac_heading_new = np.zeros(y2['Time'].shape)
            akz_lon_new = np.zeros(y2['Longitude'].shape) 
            akz_lat_new = np.zeros(y2['Latitude'].shape) 
            akaze_msi_new = np.zeros(y2['Longitude'].shape) 
            akaze_clon_new = np.zeros(y2['CornerLongitude'].shape) 
            akaze_clat_new = np.zeros(y2['CornerLatitude'].shape) 
            akaze_surfalt_new = np.zeros(y2['SurfaceAltitude'].shape)  
            akaze_Distance_Akaze_Reprojected_new = np.zeros(y2['Longitude'].shape)
            akaze_Optimization_Convergence_Fail_new = np.zeros(1)
            akaze_Reprojection_Fit_Flag_new = np.zeros(1)
            # Add avionics variables
            av_lon_new = np.zeros(y2['Longitude'].shape) 
            av_lat_new = np.zeros(y2['Latitude'].shape)  
            av_clon_new = np.zeros(y2['CornerLongitude'].shape) 
            av_clat_new = np.zeros(y2['CornerLatitude'].shape) 
            av_sza_new = np.zeros(y2['SolarZenithAngle'].shape) 
            av_saa_new = np.zeros(y2['SolarAzimuthAngle'].shape) 
            av_aza_new = np.zeros(y2['RelativeAzimuthAngle'].shape)  
            av_vza_new = np.zeros(y2['ViewingZenithAngle'].shape) 
            av_vaa_new = np.zeros(y2['ViewingAzimuthAngle'].shape)  
            av_surfalt_new = np.zeros(y2['SurfaceAltitude'].shape)  
            av_ac_lon_new = np.zeros(y2['AircraftLongitude'].shape) 
            av_ac_lat_new = np.zeros(y2['AircraftLatitude'].shape) 
            av_ac_alt_surf_new = np.zeros(y2['AircraftAltitudeAboveSurface'].shape) 
            av_ac_surf_alt_new = np.zeros(y2['AircraftSurfaceAltitude'].shape) 
            av_ac_pix_bore_new = np.zeros(y2['AircraftPixelBore'].shape) 
            av_ac_pos_new = np.zeros(y2['AircraftPos'].shape) 
            av_obsalt_new = np.zeros(y2['ObservationAltitude'].shape)  
        if(akaze_used_new==1): 

            ac_roll_new = np.zeros(y2['Time'].shape)
            ac_pitch_new = np.zeros(y2['Time'].shape)
            ac_heading_new = np.zeros(y2['Time'].shape)
            akaze_msi_new = np.zeros(y2['Longitude'].shape)
            av_lon_new = np.zeros(y2['Longitude'].shape)
            av_lat_new = np.zeros(y2['Latitude'].shape)
            av_clon_new = np.zeros(y2['CornerLongitude'].shape)
            av_clat_new = np.zeros(y2['CornerLatitude'].shape)
            av_surfalt_new = np.zeros(y2['SurfaceAltitude'].shape)
            # Add all optimized variables.shape)
            op_lon_new = np.zeros(y2['Longitude'].shape)
            op_lat_new = np.zeros(y2['Latitude'].shape)
            op_clon_new = np.zeros(y2['CornerLongitude'].shape)
            op_clat_new = np.zeros(y2['CornerLatitude'].shape)
            op_sza_new = np.zeros(y2['SolarZenithAngle'].shape)
            op_saa_new = np.zeros(y2['SolarAzimuthAngle'].shape)
            op_aza_new = np.zeros(y2['RelativeAzimuthAngle'].shape)
            op_vza_new = np.zeros(y2['ViewingZenithAngle'].shape)
            op_vaa_new = np.zeros(y2['ViewingAzimuthAngle'].shape)
            op_surfalt_new = np.zeros(y2['SurfaceAltitude'].shape)
            op_ac_lon_new = np.zeros(y2['AircraftLongitude'].shape)
            op_ac_lat_new = np.zeros(y2['AircraftLatitude'].shape)
            op_ac_alt_surf_new = np.zeros(y2['AircraftAltitudeAboveSurface'].shape)
            op_ac_surf_alt_new = np.zeros(y2['AircraftSurfaceAltitude'].shape)
            op_ac_pix_bore_new = np.zeros(y2['AircraftPixelBore'].shape)
            op_ac_pos_new = np.zeros(y2['AircraftPos'].shape)
            op_obsalt_new = np.zeros(y2['ObservationAltitude'].shape)
            akaze_Distance_Akaze_Reprojected_new = np.zeros(y2['Longitude'].shape)
            akaze_Optimization_Convergence_Fail_new = np.zeros(1)
            akaze_Reprojection_Fit_Flag_new = np.zeros(1)

        
        wvl_new.fill(np.nan)
        rad_err_new.fill(np.nan)
        rad_flags_new.fill(np.nan)
        rad_new.fill(np.nan) 
        
        corner_lon_new.fill(np.nan) 
        corner_lat_new.fill(np.nan)
        lon_new.fill(np.nan) 
        lat_new.fill(np.nan) 
        sza_new.fill(np.nan) 
        vza_new.fill(np.nan) 
        vaa_new.fill(np.nan) 
        saa_new.fill(np.nan)
        aza_new.fill(np.nan) 
        surfalt_new.fill(np.nan) 
        time_new.fill(np.nan) 
        obsalt_new.fill(np.nan)  
        time_new.fill(np.nan) 
        ac_lon_new.fill(np.nan)  
        ac_lat_new.fill(np.nan)  
        ac_alt_surf_new.fill(np.nan) 
        ac_surf_alt_new.fill(np.nan) 
        ac_pix_bore_new.fill(np.nan) 
        ac_pos_new.fill(np.nan) 
        #akaze_used_new.fill(np.nan) 
        #optimized_used_new.fill(np.nan) 
        #avionics_used_new.fill(np.nan) 
        
        if(optimized_used_new==1): 
            ac_roll_new.fill(np.nan) 
            ac_pitch_new.fill(np.nan)
            ac_heading_new.fill(np.nan)
            akz_lon_new.fill(np.nan) 
            akz_lat_new.fill(np.nan) 
            akaze_msi_new.fill(np.nan) 
            akaze_clon_new.fill(np.nan)
            akaze_clat_new.fill(np.nan)
            akaze_surfalt_new.fill(np.nan)
            akaze_Distance_Akaze_Reprojected_new.fill(np.nan)
            akaze_Optimization_Convergence_Fail_new.fill(np.nan)
            akaze_Reprojection_Fit_Flag_new.fill(np.nan)
            # Add avionics variables.fill(np.nan)
            av_lon_new.fill(np.nan)
            av_lat_new.fill(np.nan)
            av_clon_new.fill(np.nan)
            av_clat_new.fill(np.nan)
            av_sza_new.fill(np.nan)
            av_saa_new.fill(np.nan)
            av_aza_new.fill(np.nan)
            av_vza_new.fill(np.nan)
            av_vaa_new.fill(np.nan)
            av_surfalt_new.fill(np.nan)
            av_ac_lon_new.fill(np.nan)
            av_ac_lat_new.fill(np.nan)
            av_ac_alt_surf_new.fill(np.nan)
            av_ac_surf_alt_new.fill(np.nan)
            av_ac_pix_bore_new.fill(np.nan)
            av_ac_pos_new.fill(np.nan)
            av_obsalt_new.fill(np.nan)
        if(akaze_used_new==1): 

            ac_roll_new.fill(np.nan) 
            ac_pitch_new.fill(np.nan)
            ac_heading_new.fill(np.nan)
            akaze_msi_new.fill(np.nan)
            av_lon_new.fill(np.nan)
            av_lat_new.fill(np.nan)
            av_clon_new.fill(np.nan)
            av_clat_new.fill(np.nan)
            av_surfalt_new.fill(np.nan)
            # Add all optimized variables.shape).fill(np.nan)
            op_lon_new.fill(np.nan)
            op_lat_new.fill(np.nan)
            op_clon_new.fill(np.nan)
            op_clat_new.fill(np.nan)
            op_sza_new.fill(np.nan)
            op_saa_new.fill(np.nan)
            op_aza_new.fill(np.nan)
            op_vza_new.fill(np.nan)
            op_vaa_new.fill(np.nan)
            op_surfalt_new.fill(np.nan)
            op_ac_lon_new.fill(np.nan)
            op_ac_lat_new.fill(np.nan)
            op_ac_alt_surf_new.fill(np.nan)
            op_ac_surf_alt_new.fill(np.nan)
            op_ac_pix_bore_new.fill(np.nan)
            op_ac_pos_new.fill(np.nan)
            op_obsalt_new.fill(np.nan)
            akaze_Distance_Akaze_Reprojected_new.fill(np.nan)
            akaze_Optimization_Convergence_Fail_new.fill(np.nan)
            akaze_Reprojection_Fit_Flag_new.fill(np.nan)
        
        
        
        
        if(fileo2 is not None): 
            nframeso2 = np.int(rado2.shape[1] )
            nframesch4 = np.int(radch4.shape[1]) 
            datao2 = np.nanmedian(rado2[100:450,:,:],axis=0)
            datach4 = np.nanmedian(radch4[100:450,:,:],axis=0)
            
            datao2 = ((datao2/np.nanmax(datao2))*255.0)
            datach4 = ((datach4/np.nanmax(datach4))*255.0)
            
            
            plt.rcParams.update({'font.size': 8})
            
            #idxo2 = np.isfinite(rado2[600,0,:])
            idxo2 = np.isfinite((np.nanmean(np.nanmean(rado2[100:450,:,:],axis=0),axis=0)))
            #idxch4 = np.isfinite(radch4[600,0,:])
            idxch4 = np.isfinite((np.nanmean(np.nanmean(radch4[100:450,:,:],axis=0),axis=0)))
            stopch4 = False
            completech4 = False
            for i in range(1280):
                if((idxch4[i] == True) and (stopch4 == False)):
                    print("CH4 data start Xtrack = ",i)
                    ch4_start = i
                    stopch4 = True
                elif((completech4 == False) and (stopch4 == True) and (idxch4[i] == False)):
                    completech4 = True
                    ch4_end = i
                    print("CH4 data end Xtrack = ",i)
             
            stopo2 = False
            completeo2 = False
            for i in range(1280):
                if((idxo2[i] == True) and (stopo2 == False)):
                    print("O2 data start Xtrack = ",i)
                    o2_start = i
                    stopo2 = True
                elif((completeo2 == False) and (stopo2 == True) and (idxo2[i] == False)):
                    o2_end = i
                    completeo2 = True
                    print("O2 data end Xtrack = ",i)
            x = 0
            if(stopch4 == True and stopo2 == True):
            
                n = abs(nframesch4 - nframeso2)
            
                cut = n + 2 
            
            
                if((nframesch4 <= 10) and (nframesch4 >= 6)):
                    cut = np.int(1)
                    #ch4_cut = np.zeros(( (nframesch4 - np.int(2*cut)  ) ,  (datach4.shape[1] - 400 )   ))
                    #ch4_cut = datach4[cut:nframesch4-cut,(ch4_start+400):(ch4_start+400+400)]
                    ch4_cut = np.zeros(( (nframesch4 - np.int(2*cut)  ) ,  int(ch4_xlen - 100 )   ))
                    ch4_cut = datach4[np.int(1*cut):np.int(nframesch4-1*cut),(ch4_start+50):(ch4_end-50)]
                elif(nframesch4 > 10 ):
                    ch4_xlen =  int(ch4_end - ch4_start)
                    #cut = np.floor(nframesch4/80)
                    ch4_cut = np.zeros(( (nframesch4 - np.int(2*cut)  ) ,  int(ch4_xlen - 100 )   ))
                    ch4_cut = datach4[np.int(1*cut):np.int(nframesch4-1*cut),(ch4_start+50):(ch4_end-50)]
                else:
                    print("Very few frames: ",inputfile)
                    x = 1
            
                if(x == 0):  
                    o2_xlen =  int(o2_end - o2_start)
                    o2_cut = np.zeros((nframeso2   ,  o2_xlen  )  )
                    o2_cut = datao2[:,int(o2_start):int(o2_end)]
                    im = Image.fromarray(np.uint8(o2_cut),'L')
                    im1 = Image.fromarray(np.uint8(ch4_cut),'L')
                    #im = Image.fromarray(np.uint8(datao2),'L')
                    #im1 = Image.fromarray(np.uint8(o2_cut),'L')
                    
                    im.save('o2.png')
                    im1.save('ch4.png')
                    
                    #####################################################
                    
                    
                    # Read the images from the file
                    small_image = cv2.imread('ch4.png')
                    large_image = cv2.imread('o2.png')
                    
                    
                    method = cv2.TM_SQDIFF_NORMED
                    result = cv2.matchTemplate(small_image, large_image, method)
                    
                    # We want the minimum squared difference
                    mn,_,mnLoc,_ = cv2.minMaxLoc(result)
                    
                    # Draw the rectangle:
                    # Extract the coordinates of our best match
                    MPx,MPy = mnLoc
                    
                    # Step 2: Get the size of the template. This is the same size as the match.
                    trows,tcols = small_image.shape[:2]
                    
                    
                    #xshift = (ch4_start + 50)  - (MPx - o2_start) 
                    xshift = (ch4_start+50 )  -MPx 
            
                    # MPy is the location in O2 data where the "cut" CH4 frame matches O2. 
            
                    ashift = cut - MPy
                    if(abs(ashift) > 2):
                        print('Atrack Shift fails')
                        ashift = 0 
                        print('Assigning Atrack shift = 0 : the default')
                    if((abs(xshift) > 310) or (abs(xshift) < 300)):
                        print('Xtrack Shift fails')
                        print('Assigning Xtrack shift = 304 : the default')
                        xshift = 304 

                elif(x == 1):  
                    print('Very few frames:')
                    print('Assigning Atrack shift = 0 : the default')
                    print('Assigning Xtrack shift = 304 : the default')
                    ashift = 0 
                    xshift = 304 
        else:
            datach4 = np.nanmedian(radch4[100:450,:,:],axis=0)
            ashift = 0 
            xshift = 304
            nframesch4 = np.int(radch4.shape[1]) 
            nframeso2 = nframesch4
            o2_start = 142 
            o2_end = 1005
        if( (xshift >= 300) and (xshift <= 310) ): 
            #if( (xshift >= 0)):# and (xshift <= 340) ): 
                # Step 3: Draw the rectangle on large_image
                #cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)
                
                # Display the original image with the rectangle around the match.
                #cv2.imshow('output',large_image)
                
                # The image is only displayed if we call this
                #cv2.waitKey(0)
                 
                ###################################################
                #               MAKE NEW CH4 ARRAY                #
                ###################################################
                # more o2 frames means larger ch4 data with nans         
        
                print('File = ',filech4 )
                if(fileo2 is not None): 
                    print('o2 shape = ',datao2.shape)
                else: 
                    print('No Mathed O2 File')
                print('xshift = ',xshift)
                print('ashift = ',ashift)
        
        
        
                if(fileo2 is not None): 
                    ch4_new = np.zeros(( (datao2.shape[0]  ) ,  (datach4.shape[1]  )   ))
                else: 
                    ch4_new = np.zeros(( (datach4.shape[0]  ) ,  (datach4.shape[1]  )   ))
                if((ashift >= 0) and (nframeso2 >= nframesch4) ):
                    n = nframeso2-nframesch4
                    rad_err_new[:,:,0:o2_start] = np.nan
                    rad_new[:,:,0:o2_start] = np.nan
                    a1= 0
                    a2= (np.int(nframesch4)-np.int(ashift))
                    b1= np.int(o2_start)
                    b2= 1280
                    c1= np.int(ashift)
                    c2= nframesch4
                    d1= np.int(xshift)
                    d2= (1280 - np.int(o2_start) + np.int(xshift))
                      
                    if(d2 > 1280):
                        over = d2 - 1280
                        d2 = 1280
                        b2 = 1280 - over  
                      
                    if(optimized_used_new==1): 
                   
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                    elif(akaze_used_new==1): 
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                    else: 
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                   
                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
        
                    for m in range(len(self.CH4Times)):
        
                        mseconds = abs(np.int(self.CH4StartTime[m])) % 100
                        mminutes = ((abs(np.int(self.CH4StartTime[m])) % 10000) - mseconds)/100
                        mhour = (np.int(self.CH4StartTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                        if(self.delta != np.int(9)):
                            if(np.int(self.inputCH4EndTime)%10 != 9): 
                                timediff = 0 
                            else: 
                                timediff = 1 
                        else:
                            timediff = 1
                        if( (np.int(self.hseconds) == 59   ) and (np.int(self.hminutes) == 59   )  ): # change hour and minute and second of m file 
                            if( ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(self.hhour + 1)   )) or  ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(self.hhour )   ))    ):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= (np.int(nframesch4) - np.int(abs(ashift) ))
                                a2= nframesch4
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= 0
                                c2= np.int(abs(ashift))+n
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))   
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
        
                        elif( (np.int(self.hseconds) == 59   ) and (np.int(self.hminutes) != 59   )  ): #  minute and second of m file V
                            if( ((np.int(mseconds) == 00   ) and (np.int(mminutes) == (np.int(self.hminutes) + 1)   ) and (np.int(mhour) == np.int(self.hhour )   ) ) or  ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(self.hminutes) )   ) and (np.int(mhour) == np.int(self.hhour )   ) )   ):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= (np.int(nframesch4) - np.int(abs(ashift) ))
                                a2= nframeso2
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= 0
                                c2= np.int(abs(ashift))+n
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                    
                        elif( (np.int(self.hseconds) != 59   ) and (np.int(self.hminutes) == 59   )  ): #  minute and second of m file V
                            if( math.isclose(np.int(mseconds) ,np.int(self.hseconds),abs_tol=1 )    and (np.int(mminutes) == (np.int(self.hminutes) )   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= (np.int(nframesch4) - np.int(abs(ashift) ))
                                a2= nframeso2
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= 0
                                c2= np.int(abs(ashift))+n
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift)) 
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                        elif( (np.int(self.hseconds) != 59   ) and (np.int(self.hminutes) != 59   )  ): #  minute and second of m file V
                            if( math.isclose(np.int(mseconds) , np.int(self.hseconds),abs_tol=1.1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ))  :  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= (np.int(nframesch4) - np.int(abs(ashift) ))
                                a2= nframeso2
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= 0
                                c2= np.int(abs(ashift))+n
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))   
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
        
                elif((ashift >= 0) and (nframeso2 < nframesch4) ):
                    n = nframesch4 - nframeso2
                    ch4_new[:,0:o2_start] = np.nan
                    rad_err_new[:,:,0:o2_start] = np.nan
                    rad_new[:,:,0:o2_start] = np.nan
                    a1= 0
                    a2= (np.int(nframeso2)-np.int(ashift))
                    b1= np.int(o2_start)
                    b2= 1280
                    c1= np.int(ashift)
                    c2= np.int(nframeso2)
                    d1= np.int(xshift)
                    d2= (1280 - np.int(o2_start) + np.int(xshift))
                    if(d2 > 1280):
                        over = d2 - 1280
                        d2 = 1280
                        b2 = 1280 - over  
                    
                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                    if(optimized_used_new==1): 
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                    elif(akaze_used_new==1): 
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                    else: 
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                    for m in range(len(self.CH4Times)):
        
                        mseconds = abs(np.int(self.CH4StartTime[m])) % 100
                        mminutes = ((abs(np.int(self.CH4StartTime[m])) % 10000) - mseconds)/100
                        mhour = (np.int(self.CH4StartTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                        if(self.delta != np.int(9)):
                            if(np.int(self.inputCH4EndTime)%10 != 9): 
                                timediff = 0 
                            else: 
                                timediff = 1 
                        else:
                            timediff = 1
        
                        if( (np.int(self.hseconds) == 59   ) and (np.int(self.hminutes) == 59   )  ): # change hour and minute and second of m file 
                            if( ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(self.hhour + 1)   )) or  ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(self.hhour )   ))    ):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= (np.int(nframeso2) - np.int(abs(ashift) ))
                                a2= nframeso2
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= 0
                                c2= np.int(abs(ashift))
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift)) 
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                        elif( (np.int(self.hseconds) == 59   ) and (np.int(self.hminutes) != 59   )  ): #  minute and second of m file V
                            if( ((np.int(mseconds) == 00   ) and (np.int(mminutes) == (np.int(self.hminutes) + 1)   ) and (np.int(mhour) == np.int(self.hhour )   ) ) or ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(self.hminutes) )   ) and (np.int(mhour) == np.int(self.hhour )   ) )    ):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= (np.int(nframeso2) - np.int(abs(ashift) ))
                                a2= nframeso2
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= 0
                                c2= np.int(abs(ashift))
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift)) 
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                        elif( (np.int(self.hseconds) != 59   ) and (np.int(self.hminutes) == 59   )  ): #  minute and second of m file V
                            if( math.isclose(np.int(mseconds) ,(np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ))  :  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= (np.int(nframeso2) - np.int(abs(ashift) ))
                                a2= nframeso2
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= 0
                                c2= np.int(abs(ashift))
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                
                        elif( (np.int(self.hseconds) != 59   ) and (np.int(self.hminutes) != 59   )  ): #  minute and second of m file V
                            if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= (np.int(nframeso2) - np.int(abs(ashift) ))
                                a2= nframeso2
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= 0
                                c2= np.int(abs(ashift))
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                
        
                elif((ashift < 0) and (nframeso2 >= nframesch4) ):
                    n = nframeso2 - nframesch4
                    rad_err_new[:,:,0:o2_start] = np.nan
                    rad_new[:,:,0:o2_start] = np.nan
                    ch4_new[:,0:o2_start] = np.nan
        
                    if(np.int(abs(ashift)) == n):
                        a1= np.int(abs(ashift))
                        a2= (  np.int(abs(ashift)) + np.int(nframesch4))
                        b1= np.int(o2_start)
                        b2= 1280
                        c1= 0
                        c2= nframesch4
                        d1= np.int(xshift)
                        d2= (1280 - np.int(o2_start) + np.int(xshift))
                        if(d2 > 1280):
                            over = d2 - 1280
                            d2 = 1280
                            b2 = 1280 - over  
                        
                        ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                        if(optimized_used_new==1): 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        elif(akaze_used_new==1): 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        else: 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        # find previous ch4 granule last #|ashift| frames... 
                        for m in range(len(self.CH4Times)):
        
                            mseconds = abs(np.int(self.CH4EndTime[m])) % 100
                            mminutes = ((abs(np.int(self.CH4EndTime[m])) % 10000) - mseconds)/100
                            mhour = (np.int(self.CH4EndTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                            if(self.delta != np.int(9)):
                                if(np.int(self.inputCH4StartTime)%10 != 0): 
                                    timediff = 0 
                                else: 
                                    timediff = 1 
        
                            else:
                                timediff = 1
                            if( (np.int(self.hseconds) == 0   ) and (np.int(self.hminutes) == 0   )  ): # change hour and minute and second of m file 
                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(self.hhour - 1)   )) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(self.hhour )   ))    ):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift)) 
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    
        
                            elif( (np.int(self.hseconds) == 0   ) and (np.int(self.hminutes) != 0   )  ): #  minute and second of m file V
                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(self.hminutes) -1)   ) and (np.int(mhour) == np.int(self.hhour )   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == (np.int(self.hminutes) )   ) and (np.int(mhour) == np.int(self.hhour )   ) ) ):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift)) 
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    
        
                            elif( (np.int(self.hseconds) != 0   ) and (np.int(self.hminutes) == 0   )  ): #  minute and second of m file V
                                if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    
                            elif( (np.int(self.hseconds) != 0   ) and (np.int(self.hminutes) != 0   )  ): #  minute and second of m file V
                                if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   )) :  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))  
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
        
                    elif(np.int(abs(ashift)) < n):
                        a1= np.int(abs(ashift))
                        a2= (  np.int(abs(ashift)) + np.int(nframesch4))
                        b1= np.int(o2_start)
                        b2= 1280
                        c1= 0
                        c2= nframesch4
                        d1= np.int(xshift)
                        d2= (1280 - np.int(o2_start) + np.int(xshift))
                        if(d2 > 1280):
                            over = d2 - 1280
                            d2 = 1280
                            b2 = 1280 - over  
                        
                        ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                        if(optimized_used_new==1): 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        elif(akaze_used_new==1): 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        else: 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        # find previous ch4 granule last #|ashift| frames... 
                        for m in range(len(self.CH4Times)):
        
                            mseconds = abs(np.int(self.CH4EndTime[m])) % 100
                            mminutes = ((abs(np.int(self.CH4EndTime[m])) % 10000) - mseconds)/100
                            mhour = (np.int(self.CH4EndTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                            if(self.delta != np.int(9)):
                                if(np.int(self.inputCH4StartTime)%10 != 0): 
                                    timediff = 0 
                                else: 
                                    timediff = 1 
        
                            else:
                                timediff = 1
                            if( (np.int(self.hseconds) == 0   ) and (np.int(self.hminutes) == 0   )  ): # change hour and minute and second of m file 
                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(self.hhour - 1)   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(self.hhour )   ) )):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift)) 
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    
        
                            elif( (np.int(self.hseconds) == 0   ) and (np.int(self.hminutes) != 0   )  ): #  minute and second of m file V
                                if(( (np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(self.hminutes) -1)   ) and (np.int(mhour) == np.int(self.hhour )   ) ) or ( (np.int(mseconds) == 0   ) and (np.int(mminutes) == (np.int(self.hminutes) )   ) and (np.int(mhour) == np.int(self.hhour )   ) )):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift)) 
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                            elif( (np.int(self.hseconds) != 0   ) and (np.int(self.hminutes) == 0   )  ): #  minute and second of m file V
                                if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                            elif( (np.int(self.hseconds) != 0   ) and (np.int(self.hminutes) != 0   )  ): #  minute and second of m file V
        
                                if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))  
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
        
        
                        for m in range(len(self.CH4Times)):
        
                            mseconds = abs(np.int(self.CH4StartTime[m])) % 100
                            mminutes = ((abs(np.int(self.CH4StartTime[m])) % 10000) - mseconds)/100
                            mhour = (np.int(self.CH4StartTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                            if(self.delta != np.int(9)):
                                if(np.int(self.inputCH4EndTime)%10 != 9): 
                                    timediff = 0 
                                else: 
                                    timediff = 1 
                            else:
                                timediff = 1
                            if( (np.int(self.hseconds) == 59   ) and (np.int(self.hminutes) == 59   )  ): # change hour and minute and second of m file 
                                if( ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(self.hhour + 1)   ) ) or  ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(self.hhour )   ) )):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= (np.int(nframeso2) - np.int(abs(n)) + np.int(abs(ashift) ))
                                    a2= nframeso2
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= 0
                                    c2= -np.int(abs(ashift))+n
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))   
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                            elif( (np.int(self.hseconds) == 59   ) and (np.int(self.hminutes) != 59   )  ): #  minute and second of m file V
                                if( ((np.int(mseconds) == 00   ) and (np.int(mminutes) == (np.int(self.hminutes) + 1)   ) and (np.int(mhour) == np.int(self.hhour )   ) ) or((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(self.hminutes) )   ) and (np.int(mhour) == np.int(self.hhour )   ) )):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= (np.int(nframeso2) - np.int(abs(n)) + np.int(abs(ashift) ))
                                    a2= nframeso2
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= 0
                                    c2= -np.int(abs(ashift))+n
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))   
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        
                            elif( (np.int(self.hseconds) != 59   ) and (np.int(self.hminutes) == 59   )  ): #  minute and second of m file V
                                if( math.isclose(np.int(mseconds) ,  (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes) )   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                    
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= (np.int(nframeso2) - np.int(abs(n)) + np.int(abs(ashift) ))
                                    a2= nframeso2
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= 0
                                    c2= -np.int(abs(ashift))+n
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))   
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                            elif( (np.int(self.hseconds) != 59   ) and (np.int(self.hminutes) != 59   )  ): #  minute and second of m file V
                                if( math.isclose(np.int(mseconds) == (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= (np.int(nframeso2) - np.int(abs(n)) + np.int(abs(ashift) ))
                                    a2= nframeso2
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= 0
                                    c2= -np.int(abs(ashift))+n
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))   
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    
        
                    else:
                        a1=np.int(abs(ashift))
                        a2=( np.int(nframesch4) + n  )
                        b1=np.int(o2_start)
                        b2=1280
                        c1=0 
                        c2=(np.int(nframesch4) - np.int(abs(ashift)) + n) 
                        d1=np.int(xshift) 
                        d2=(1280 - np.int(o2_start) + np.int(xshift))
                        if(d2 > 1280):
                            over = d2 - 1280
                            d2 = 1280
                            b2 = 1280 - over  
                        
                        ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                        if(optimized_used_new==1): 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        elif(akaze_used_new==1): 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        else: 
                            rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
        
                        for m in range(len(self.CH4Times)):
        
                            mseconds = abs(np.int(self.CH4EndTime[m])) % 100
                            mminutes = ((abs(np.int(self.CH4EndTime[m])) % 10000) - mseconds)/100
                            mhour = (np.int(self.CH4EndTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                            if(self.delta != np.int(9)):
                                if(np.int(self.inputCH4StartTime)%10 != 0): 
                                    timediff = 0 
                                else: 
                                    timediff = 1 
        
                            else:
                                timediff = 1
                            if( (np.int(self.hseconds) == 0   ) and (np.int(self.hminutes) == 0   )  ): # change hour and minute and second of m file 
                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(self.hhour - 1)   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(self.hhour )   ) )):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1=0  
                                    a2=np.int(abs(ashift))  
                                    b1=np.int(o2_start)  
                                    b2=1280  
                                    c1=(new_nframesch4-np.int(abs(ashift)))  
                                    c2=np.int(new_nframesch4)  
                                    d1=np.int(xshift)  
                                    d2=(1280 - np.int(o2_start) + np.int(xshift))     
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                            elif( (np.int(self.hseconds) == 0   ) and (np.int(self.hminutes) != 0   )  ): #  minute and second of m file V
                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(self.hminutes) -1)   ) and (np.int(mhour) == np.int(self.hhour )   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == (np.int(self.hminutes) )   ) and (np.int(mhour) == np.int(self.hhour )   ) )):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1=0 
                                    a2=np.int(abs(ashift)) 
                                    b1=np.int(o2_start) 
                                    b2=1280 
                                    c1=(new_nframesch4-np.int(abs(ashift))) 
                                    c2=np.int(new_nframesch4) 
                                    d1=np.int(xshift) 
                                    d2=(1280 - np.int(o2_start) + np.int(xshift)) 
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    
        
                            elif( (np.int(self.hseconds) != 0   ) and (np.int(self.hminutes) == 0   )  ): #  minute and second of m file V
                                if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                            elif( (np.int(self.hseconds) != 0   ) and (np.int(self.hminutes) != 0   )  ): #  minute and second of m file V
                                if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                    add_filech4 = self.CH4Files[m] 
                                    new_datach4 = Dataset(add_filech4)
                                    self.x1 = new_datach4.groups['Band1']
                                    self.x2 = new_datach4.groups['Geolocation']
                                    add_radch4 = self.x1['Radiance'][:,:,:].data
                                    new_nframesch4 = add_radch4.shape[1]
                                    add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                    add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                    a1= 0
                                    a2= np.int(abs(ashift))
                                    b1= np.int(o2_start)
                                    b2= 1280
                                    c1= (new_nframesch4-np.int(abs(ashift)))
                                    c2= new_nframesch4
                                    d1= np.int(xshift)
                                    d2= (1280 - np.int(o2_start) + np.int(xshift))
                                    if(d2 > 1280):
                                        over = d2 - 1280
                                        d2 = 1280
                                        b2 = 1280 - over  
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    if(optimized_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    elif(akaze_used_new==1): 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    else: 
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                                                           
                elif((ashift < 0) and (nframeso2 < nframesch4) ):
                    n = nframesch4 - nframeso2
                    ch4_new[:,0:o2_start] = np.nan
                    rad_err_new[:,:,0:o2_start] = np.nan
                    rad_new[:,:,0:o2_start] = np.nan
                    a1= np.int(abs(ashift))
                    a2= (  np.int(nframeso2))
                    b1= np.int(o2_start)
                    b2= 1280
                    c1= 0
                    c2= ( np.int(nframeso2) - np.int(abs(ashift))    )
                    d1= np.int(xshift)
                    d2= (1280 - np.int(o2_start) + np.int(xshift))
                    if(d2 > 1280):
                        over = d2 - 1280
                        d2 = 1280
                        b2 = 1280 - over  
                    
                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                    if(optimized_used_new==1): 
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                    elif(akaze_used_new==1): 
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                    else: 
                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                    
                    for m in range(len(self.CH4Times)):
        
                        mseconds = abs(np.int(self.CH4EndTime[m])) % 100
                        mminutes = ((abs(np.int(self.CH4EndTime[m])) % 10000) - mseconds)/100
                        mhour = (np.int(self.CH4EndTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                        if(self.delta != np.int(9)):
                            if(np.int(self.inputCH4StartTime)%10 != 0): 
                                timediff = 0 
                            else: 
                                timediff = 1 
        
                        else:
                            timediff = 1
                        if( (np.int(self.hseconds) == 0   ) and (np.int(self.hminutes) == 0   )  ): # change hour and minute and second of m file 
                            if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(self.hhour - 1)   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(self.hhour )   ) )):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= 0
                                a2= np.int(abs(ashift))
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= (new_nframesch4-np.int(abs(ashift)))
                                c2= new_nframesch4
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                
        
                        elif( (np.int(self.hseconds) == 0   ) and (np.int(self.hminutes) != 0   )  ): #  minute and second of m file V
                            if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(self.hminutes) -1)   ) and (np.int(mhour) == np.int(self.hhour )   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) )):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= 0
                                a2= np.int(abs(ashift))
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= (new_nframesch4-np.int(abs(ashift)))
                                c2= new_nframesch4
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                        elif( (np.int(self.hseconds) != 0   ) and (np.int(self.hminutes) == 0   )  ): #  minute and second of m file V
                            if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= 0
                                a2= np.int(abs(ashift))
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= (new_nframesch4-np.int(abs(ashift)))
                                c2= new_nframesch4
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))
                                
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                        elif( (np.int(self.hseconds) != 0   ) and (np.int(self.hminutes) != 0   )  ): #  minute and second of m file V
                            if( math.isclose(np.int(mseconds) , (np.int(self.hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(self.hminutes))   ) and (np.int(mhour) == np.int(self.hhour )   ) ):  
                                add_filech4 = self.CH4Files[m] 
                                new_datach4 = Dataset(add_filech4)
                                self.x1 = new_datach4.groups['Band1']
                                self.x2 = new_datach4.groups['Geolocation']
                                add_radch4 = self.x1['Radiance'][:,:,:].data
                                new_nframesch4 = add_radch4.shape[1]
                                add_datach4 = np.nanmedian(add_radch4[100:450,:,:],axis=0)
                                add_datach4 = ((add_datach4/np.nanmax(add_datach4))*255.0)
                                a1= 0
                                a2= np.int(abs(ashift))
                                b1= np.int(o2_start)
                                b2= 1280
                                c1= (new_nframesch4-np.int(abs(ashift)))
                                c2= new_nframesch4
                                d1= np.int(xshift)
                                d2= (1280 - np.int(o2_start) + np.int(xshift))
                                if(d2 > 1280):
                                    over = d2 - 1280
                                    d2 = 1280
                                    b2 = 1280 - over  
                                
                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                
                                if(optimized_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_optimized(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akz_lon_new,akz_lat_new,akaze_msi_new,akaze_clon_new,akaze_clat_new,akaze_surfalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_sza_new,av_saa_new,av_aza_new,av_vza_new,av_vaa_new,av_surfalt_new,av_ac_lon_new,av_ac_lat_new,av_ac_alt_surf_new,av_ac_surf_alt_new,av_ac_pix_bore_new,av_ac_pos_new,av_obsalt_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                elif(akaze_used_new==1): 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new = self.assign_data_akaze(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,akaze_msi_new,av_lon_new,av_lat_new,av_clon_new,av_clat_new,av_surfalt_new,op_lon_new,op_lat_new,op_clon_new,op_clat_new,op_sza_new,op_saa_new,op_aza_new,op_vza_new,op_vaa_new,op_surfalt_new,op_ac_lon_new,op_ac_lat_new,op_ac_alt_surf_new,op_ac_surf_alt_new,op_ac_pix_bore_new,op_ac_pos_new,op_obsalt_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new,akaze_used_new,optimized_used_new,avionics_used_new,ac_roll_new,ac_pitch_new,ac_heading_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                else: 
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                    
                    
        
                else:
                    print("Missed Something!")
                    exit()
                if(fileo2 is not None):
                    total_shift_in_pixels = xshift - 110 - 50 + MPx 
                else:
                    total_shift_in_pixels = xshift - 110 - 50 + 29 
                if(fileo2 is not None):
                    #############################
                    #  NANS APPLIED TO CH4 DATA FROM O2 DATA POSSESSING NANS 
                    #############################
                    t = np.nanmean(rad_new[100:450,:,:],axis=0)
                    valpix = np.isfinite(np.nanmean(y1['Radiance'][100:450,:,:],axis=0))
                    ch4_new[valpix!=True] = np.nan
        
                    valpix = np.isfinite(y1['Radiance'][:,:,:])
        
                    #wvl_new[valpix!=True] = np.nan 
                    rad_err_new[valpix!=True] = np.nan
                    rad_flags_new = rad_flags_new.astype(int)
                    rad_flags_new[valpix!=True] = np.int(0)
                    rad_new[valpix!=True] = np.nan 
        
                    #############################
                    #############################
        
                    #############################
        
                    #############################
                    #  NANS APPLIED TO O2 DATA FROM CH4 DATA POSSESSING NANS 
                    #############################
                    valpix = np.isfinite(rad_new)
        
                    wvl_o2 = y1['Wavelength'][:,:,:] 
                    rad_err_o2 = y1['RadianceUncertainty'][:,:,:]
                    rad_flags_o2 = y1['RadianceFlag'][:,:,:]
                    rad_o2 = y1['Radiance'][:,:,:] 
        
                    #wvl_o2[valpix!=True] = np.nan 
                    rad_err_o2[valpix!=True] = np.nan
                    rad_flags_o2[valpix!=True] = np.int(0)
                    rad_o2[valpix!=True] = np.nan 
        
                    #############################
                    #############################
        
                    #valpix = np.isfinite(corner_lon_new)
        
                    corner_lon_o2 = y2['CornerLongitude'][:,:,:] 
                    corner_lat_o2 = y2['CornerLatitude'][:,:,:] 
                    lon_o2 = y2['Longitude'][:,:] 
                    lat_o2 = y2['Latitude'][:,:] 
                    sza_o2 = y2['SolarZenithAngle'][:,:] 
                    saa_o2 = y2['SolarAzimuthAngle'][:,:] 
                    aza_o2 = y2['RelativeAzimuthAngle'][:,:] 
                    vza_o2 = y2['ViewingZenithAngle'][:,:] 
                    vaa_o2 = y2['ViewingAzimuthAngle'][:,:] 
                    surfalt_o2 = y2['SurfaceAltitude'][:,:] 
                    ac_lon_o2 = y2['AircraftLongitude'][:]
                    ac_lat_o2 = y2['AircraftLatitude'][:]
                    ac_alt_surf_o2 = y2['AircraftAltitudeAboveSurface'][:]
                    ac_surf_alt_o2 = y2['AircraftSurfaceAltitude'][:]
                    ac_pix_bore_o2 = y2['AircraftPixelBore'][:,:,:]
                    ac_pos_o2 = y2['AircraftPos'][:,:]
        
                    if(optimized_used == 1):       
                        ac_roll_o2 = y3['AircraftRoll'][:] 
                        ac_pitch_o2 = y3['AircraftPitch'][:] 
                        ac_heading_o2 = y3['AircraftHeading'][:] 
                        akaze_lon_o2 = y3['AkazeLongitude'][:] 
                        akaze_lat_o2 = y3['AkazeLatitude'][:] 
                        akaze_msi_o2 = y3['AkazeMSIImage'][:] 
                        akaze_clon_o2 = y3['AkazeCornerLongitude'][:] 
                        akaze_clat_o2 = y3['AkazeCornerLatitude'][:] 
                        akaze_surfalt_o2 = y3['AkazeSurfaceAltitude'][:]  
                        akaze_Distance_Akaze_Reprojected_o2 = y3['DistanceAkazeReprojected'][:]  
                        akaze_Optimization_Convergence_Fail_o2 = y3['OptimizationConvergenceFail'][:] 
                        akaze_Reprojection_Fit_Flag_o2 = y3['ReprojectionFitFlag'][:] 
                        # Add avionics variables
                        av_lon_o2 = y3['AvionicsLongitude'][:] 
                        av_lat_o2 = y3['AvionicsLatitude'][:]  
                        av_clon_o2 = y3['AvionicsCornerLongitude'][:] 
                        av_clat_o2 = y3['AvionicsCornerLatitude'][:] 
                        av_sza_o2 = y3['AvionicsSolarZenithAngle'][:] 
                        av_saa_o2 = y3['AvionicsSolarAzimuthAngle'][:] 
                        av_aza_o2 = y3['AvionicsRelativeAzimuthAngle'][:]  
                        av_vza_o2 = y3['AvionicsViewingZenithAngle'][:] 
                        av_vaa_o2 = y3['AvionicsViewingAzimuthAngle'][:]  
                        av_surfalt_o2 = y3['AvionicsSurfaceAltitude'][:]  
                        av_ac_lon_o2 = y3['AvionicsAircraftLongitude'][:] 
                        av_ac_lat_o2 = y3['AvionicsAircraftLatitude'][:] 
                        av_ac_alt_surf_o2 = y3['AvionicsAircraftAltitudeAboveSurface'][:] 
                        av_ac_surf_alt_o2 = y3['AvionicsAircraftSurfaceAltitude'][:] 
                        av_ac_pix_bore_o2 = y3['AvionicsAircraftPixelBore'][:] 
                        av_ac_pos_o2 = y3['AvionicsAircraftPos'][:] 
                        av_obsalt_o2 = y3['AvionicsObservationAltitude'][:]  
 
                    elif(akaze_used == 1):       
                        ac_roll_o2 = y3['AircraftRoll'][:] 
                        ac_pitch_o2 = y3['AircraftPitch'][:] 
                        ac_heading_o2 = y3['AircraftHeading'][:] 
                        akaze_msi_o2 = y3['AkazeMSIImage'][:]
                        av_lon_o2 = y3['AvionicsLongitude'][:]
                        av_lat_o2 = y3['AvionicsLatitude'][:]
                        av_clon_o2 = y3['AvionicsCornerLongitude'][:]
                        av_clat_o2 = y3['AvionicsCornerLatitude'][:]
                        av_surfalt_o2 = y3['AvionicsSurfaceAltitude'][:]
                        # Add all optimized variables[:]
                        op_lon_o2 = y3['OptimizedLongitude'][:]
                        op_lat_o2 = y3['OptimizedLatitude'][:]
                        op_clon_o2 = y3['OptimizedCornerLongitude'][:]
                        op_clat_o2 = y3['OptimizedCornerLatitude'][:]
                        op_sza_o2 = y3['OptimizedSolarZenithAngle'][:]
                        op_saa_o2 = y3['OptimizedSolarAzimuthAngle'][:]
                        op_aza_o2 = y3['OptimizedRelativeAzimuthAngle'][:]
                        op_vza_o2 = y3['OptimizedViewingZenithAngle'][:]
                        op_vaa_o2 = y3['OptimizedViewingAzimuthAngle'][:]
                        op_surfalt_o2 = y3['OptimizedSurfaceAltitude'][:]
                        op_ac_lon_o2 = y3['OptimizedAircraftLongitude'][:]
                        op_ac_lat_o2 = y3['OptimizedAircraftLatitude'][:]
                        op_ac_alt_surf_o2 = y3['OptimizedAircraftAltitudeAboveSurface'][:]
                        op_ac_surf_alt_o2 = y3['OptimizedAircraftSurfaceAltitude'][:]
                        op_ac_pix_bore_o2 = y3['OptimizedAircraftPixelBore'][:]
                        op_ac_pos_o2 = y3['OptimizedAircraftPos'][:]
                        op_obsalt_o2 = y3['OptimizedObservationAltitude'][:]
                        akaze_Distance_Akaze_Reprojected_o2 = y3['DistanceAkazeReprojected'][:]
                        akaze_Optimization_Convergence_Fail_o2 = y3['OptimizationConvergenceFail'] [:]
                        akaze_Reprojection_Fit_Flag_o2 = y3['ReprojectionFitFlag'][:]
                    else:       
                        akaze_msi_o2 = None
                    #############################
                    #############################
                    valpix = np.isfinite(time_new)
        
                    obsalt_o2 = y2['ObservationAltitude'][:] 
                    time_o2 = y2['Time'][:] 
        
                else:
         
                    ch4_new[:,0:o2_start] = np.nan
                    ch4_new[:,o2_end:] = np.nan
                     
        
                    #wvl_new[valpix!=True] = np.nan 
                    rad_err_new[:,:,0:o2_start] = np.nan
                    rad_err_new[:,:,o2_end:] = np.nan
                    rad_flags_new = rad_flags_new.astype(int)
                    rad_flags_new[:,:,0:o2_start] = np.int(0)
                    rad_flags_new[:,:,o2_end:] = np.int(0)
                    rad_new[:,:,0:o2_start] = np.nan 
                    rad_new[:,:,o2_end:] = np.nan 
        
                #############################
                # NOW WE WRITE THE NEW FILE TO DESK: CH4 FIRST
                #############################
                xtrk_aggfac = 5
                atrk_aggfac = 1
                filename = filech4.split(".nc")[0]
                filename = filename.split("CH4_NATIVE/")[1]
                l1b_ch4_dir = os.path.join(self.dest,'CH4_5x1_Aligned/') 
                l1_outfile = os.path.join(l1b_ch4_dir+filename+'.nc')
                logfile_ch4 = os.path.join(l1b_ch4_dir+'log_file_ch4_aligned.txt') 
                #l1_outfile = str(filename)+'.nc'
                
                norm_1d = block_reduce(np.ones(obsalt_new.shape),block_size=(atrk_aggfac,),func=np.mean)
                obsalt_new = block_reduce(obsalt_new,block_size=(atrk_aggfac,),func=np.mean) ; obsalt_new = obsalt_new / norm_1d
                time_new = block_reduce(time_new,block_size=(atrk_aggfac,),func=np.mean) ; time_new = time_new / norm_1d
                ac_alt_surf_new = block_reduce(ac_alt_surf_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_alt_surf_new = ac_alt_surf_new / norm_1d
                ac_surf_alt_new = block_reduce(ac_surf_alt_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_surf_alt_new = ac_surf_alt_new / norm_1d
                ac_lon_new= block_reduce(ac_lon_new,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lon_new = ac_lon_new / norm_1d
                ac_lat_new= block_reduce(ac_lat_new,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lat_new = ac_lat_new / norm_1d
                if(akaze_used_new==1):
                   ac_roll_new    = block_reduce(ac_roll_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll_new = ac_roll_new / norm_1d 
                   ac_pitch_new   = block_reduce(ac_pitch_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch_new = ac_pitch_new / norm_1d 
                   ac_heading_new = block_reduce(ac_heading_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading_new = ac_heading_new / norm_1d 
                   op_ac_lon_new  = block_reduce(op_ac_lon_new,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lon_new = op_ac_lon_new / norm_1d
                   op_ac_lat_new  = block_reduce(op_ac_lat_new,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lat_new = op_ac_lat_new / norm_1d
                   op_ac_alt_surf_new  = block_reduce(op_ac_alt_surf_new,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_alt_surf_new = op_ac_alt_surf_new/ norm_1d
                   op_ac_surf_alt_new  = block_reduce(op_ac_surf_alt_new,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_surf_alt_new = op_ac_surf_alt_new/ norm_1d
                   op_obsalt_new  = block_reduce(op_obsalt_new,block_size=(atrk_aggfac,),func=np.mean) ; op_obsalt_new = op_obsalt_new / norm_1d
                elif(optimized_used_new==1):
                   ac_roll_new    = block_reduce(ac_roll_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll_new = ac_roll_new / norm_1d 
                   ac_pitch_new   = block_reduce(ac_pitch_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch_new = ac_pitch_new / norm_1d 
                   ac_heading_new = block_reduce(ac_heading_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading_new = ac_heading_new / norm_1d 
                   av_ac_lon_new = block_reduce(av_ac_lon_new,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lon_new  =av_ac_lon_new/ norm_1d
                   av_ac_lat_new = block_reduce(av_ac_lat_new,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lat_new  =av_ac_lat_new/ norm_1d
                   av_ac_alt_surf_new = block_reduce(av_ac_alt_surf_new,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_alt_surf_new=av_ac_alt_surf_new    / norm_1d
                   av_ac_surf_alt_new = block_reduce(av_ac_surf_alt_new,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_surf_alt_new=av_ac_surf_alt_new    / norm_1d
                   av_obsalt_new = block_reduce(av_obsalt_new,block_size=(atrk_aggfac,),func=np.mean) ;av_obsalt_new =av_obsalt_new    / norm_1d
        
                norm_2d = block_reduce(np.ones(lon_new.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
                lon_new = block_reduce(lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lon_new = lon_new / norm_2d
                lat_new = block_reduce(lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lat_new = lat_new / norm_2d
                saa_new=block_reduce(saa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  saa_new = saa_new / norm_2d
                sza_new=block_reduce(sza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  sza_new = sza_new / norm_2d
                vza_new=block_reduce(vza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vza_new = vza_new / norm_2d
                vaa_new=block_reduce(vaa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vaa_new = vaa_new / norm_2d
                aza_new=block_reduce(aza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  aza_new = aza_new / norm_2d
                surfalt_new=block_reduce(surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  surfalt_new = surfalt_new / norm_2d
       

 
                lon_new = lon_new.transpose((1,0))
                lat_new = lat_new.transpose((1,0))
                saa_new=saa_new.transpose((1,0))
                sza_new=sza_new.transpose((1,0))
                vza_new=vza_new.transpose((1,0))
                vaa_new=vaa_new.transpose((1,0))
                aza_new=aza_new.transpose((1,0))
                surfalt_new=surfalt_new.transpose((1,0))
        
        
                norm_3d = block_reduce(np.ones(ac_pix_bore_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                ac_pix_bore_new=block_reduce(ac_pix_bore_new,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ;  ac_pix_bore_new = ac_pix_bore_new / norm_3d
        
                norm_2d = block_reduce(np.ones(ac_pos_new.shape),(1,atrk_aggfac),func=np.mean)
                ac_pos_new=block_reduce(ac_pos_new,block_size=(1,atrk_aggfac),func=np.mean) ;  ac_pos_new = ac_pos_new / norm_2d
        
        
                norm_3d = block_reduce(np.ones(corner_lat_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                corner_lat_new = block_reduce(corner_lat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lat_new = corner_lat_new / norm_3d
                corner_lon_new = block_reduce(corner_lon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lon_new = corner_lon_new / norm_3d
        
                corner_lat_new = corner_lat_new.transpose((2,1,0))
                corner_lon_new = corner_lon_new.transpose((2,1,0)) 
                if(optimized_used_new==1):
                    norm_2d = block_reduce(np.ones(akz_lon_new.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
                    akz_lon_new = block_reduce(akz_lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akz_lon_new = akz_lon_new / norm_2d
                    akz_lat_new = block_reduce(akz_lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akz_lat_new = akz_lat_new / norm_2d
                    akaze_msi_new = block_reduce(akaze_msi_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi_new = akaze_msi_new / norm_2d
                    akaze_surfalt_new = block_reduce(akaze_surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_surfalt_new = akaze_surfalt_new / norm_2d 
                    akaze_Distance_Akaze_Reprojected_new = block_reduce(akaze_Distance_Akaze_Reprojected_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new / norm_2d

                    av_lon_new = block_reduce(av_lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon_new=av_lon_new   / norm_2d
                    av_lat_new = block_reduce(av_lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat_new=av_lat_new   / norm_2d
                    av_sza_new = block_reduce(av_sza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_sza_new=av_sza_new   / norm_2d
                    av_saa_new = block_reduce(av_saa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_saa_new=av_saa_new   / norm_2d
                    av_aza_new = block_reduce(av_aza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_aza_new=av_aza_new   / norm_2d
                    av_vza_new = block_reduce(av_vza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vza_new=av_vza_new   / norm_2d
                    av_vaa_new = block_reduce(av_vaa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vaa_new=av_vaa_new   / norm_2d
                    av_surfalt_new = block_reduce(av_surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt_new=av_surfalt_new/norm_2d 
                    av_lon_new = av_lon_new.transpose(1,0) 
                    av_lat_new = av_lat_new.transpose(1,0) 
                    av_sza_new = av_sza_new.transpose(1,0) 
                    av_saa_new = av_saa_new.transpose(1,0) 
                    av_aza_new = av_aza_new.transpose(1,0) 
                    av_vza_new = av_vza_new.transpose(1,0) 
                    av_vaa_new = av_vaa_new.transpose(1,0) 
                    av_surfalt_new = av_surfalt_new.transpose(1,0) 


                    norm_3d = block_reduce(np.ones(akaze_clon_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    akaze_clon_new = block_reduce(akaze_clon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clon_new = akaze_clon_new / norm_3d 
                    akaze_clat_new = block_reduce(akaze_clat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clat_new = akaze_clat_new / norm_3d 
                    av_clon_new = block_reduce(av_clon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon_new = av_clon_new / norm_3d 
                    av_clat_new = block_reduce(av_clat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat_new = av_clat_new / norm_3d 
                    av_clon_new = av_clon_new.transpose((2,1,0))
                    av_clat_new = av_clat_new.transpose((2,1,0))
                    

                    akz_lon_new = akz_lon_new.transpose(1,0)
                    akz_lat_new = akz_lat_new.transpose(1,0)
                    akaze_msi_new = akaze_msi_new.transpose(1,0)
                    akaze_surfalt_new = akaze_surfalt_new.transpose(1,0)
                    akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new.transpose(1,0)
                    akaze_clon_new = akaze_clon_new.transpose((2,1,0))
                    akaze_clat_new = akaze_clat_new.transpose((2,1,0))

                    norm_3d = block_reduce(np.ones(av_ac_pix_bore_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    av_ac_pix_bore_new = block_reduce(av_ac_pix_bore_new,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_ac_pix_bore_new = av_ac_pix_bore_new/norm_3d 

                    norm_2d = block_reduce(np.ones(av_ac_pos_new.shape),(1,atrk_aggfac),func=np.mean)
                    av_ac_pos_new = block_reduce(av_ac_pos_new,block_size=(1,atrk_aggfac),func=np.mean) ;  av_ac_pos_new = av_ac_pos_new / norm_2d 

                    av_ac_pos_new=av_ac_pos_new.transpose((1,0))
                    av_ac_pix_bore_new=av_ac_pix_bore_new.transpose((2,1,0))

                if(akaze_used_new==1):
                    norm_2d = block_reduce(np.ones(akaze_msi_new.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
                    akaze_msi_new = block_reduce(akaze_msi_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi_new = akaze_msi_new / norm_2d
                    akaze_Distance_Akaze_Reprojected_new = block_reduce(akaze_Distance_Akaze_Reprojected_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new / norm_2d
                    av_lon_new = block_reduce(av_lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon_new=av_lon_new   / norm_2d
                    av_lat_new = block_reduce(av_lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat_new=av_lat_new   / norm_2d
                    av_surfalt_new = block_reduce(av_surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt_new=av_surfalt_new/norm_2d 
                    av_lon_new = av_lon_new.transpose(1,0) 
                    av_lat_new = av_lat_new.transpose(1,0) 
                    av_surfalt_new = av_surfalt_new.transpose(1,0) 

                    op_lon_new = block_reduce(op_lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lon_new=op_lon_new  / norm_2d
                    op_lat_new = block_reduce(op_lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lat_new=op_lat_new  / norm_2d
                    op_sza_new = block_reduce(op_sza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_sza_new=op_sza_new  / norm_2d
                    op_saa_new = block_reduce(op_saa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_saa_new=op_saa_new  / norm_2d
                    op_aza_new = block_reduce(op_aza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_aza_new=op_aza_new  / norm_2d
                    op_vza_new = block_reduce(op_vza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vza_new=op_vza_new  / norm_2d
                    op_vaa_new = block_reduce(op_vaa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vaa_new=op_vaa_new  / norm_2d
                    op_surfalt_new = block_reduce(op_surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_surfalt_new =op_surfalt_new / norm_2d

                    op_lon_new = op_lon_new.transpose(1,0) 
                    op_lat_new = op_lat_new.transpose(1,0) 
                    op_sza_new = op_sza_new.transpose(1,0) 
                    op_saa_new = op_saa_new.transpose(1,0) 
                    op_aza_new = op_aza_new.transpose(1,0) 
                    op_vza_new = op_vza_new.transpose(1,0) 
                    op_vaa_new = op_vaa_new.transpose(1,0) 
                    op_surfalt_new = op_surfalt_new.transpose(1,0) 
                    akaze_msi_new = akaze_msi_new.transpose(1,0)
                    akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new.transpose(1,0)



                    norm_3d = block_reduce(np.ones(av_clon_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)

                    av_clon_new = block_reduce(av_clon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon_new = av_clon_new / norm_3d 
                    av_clat_new = block_reduce(av_clat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat_new = av_clat_new / norm_3d 
                    op_clon_new = block_reduce(op_clon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clon_new = op_clon_new / norm_3d 
                    op_clat_new = block_reduce(op_clat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clat_new = op_clat_new / norm_3d 
                    op_clon_new = op_clon_new.transpose((2,1,0))
                    op_clat_new = op_clat_new.transpose((2,1,0))
                    av_clon_new = av_clon_new.transpose((2,1,0))
                    av_clat_new = av_clat_new.transpose((2,1,0))

                    norm_3d = block_reduce(np.ones(op_ac_pix_bore_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    op_ac_pix_bore_new = block_reduce(op_ac_pix_bore_new,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_ac_pix_bore_new = op_ac_pix_bore_new/norm_3d 

                    norm_2d = block_reduce(np.ones(op_ac_pos_new.shape),(1,atrk_aggfac),func=np.mean)
                    op_ac_pos_new = block_reduce(op_ac_pos_new,block_size=(1,atrk_aggfac),func=np.mean) ;  op_ac_pos_new = op_ac_pos_new / norm_2d 

                    op_ac_pos_new=op_ac_pos_new.transpose((1,0))
                    op_ac_pix_bore_new=op_ac_pix_bore_new.transpose((2,1,0))
        
                norm_3d = block_reduce(np.ones(rad_new.shape),block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
                valpix = np.zeros(rad_new.shape)
                idv = np.logical_and(np.isfinite(rad_new),rad_new>0.0)
                valpix[idv] = 1.0
                valpix_agg = block_reduce(valpix,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
                valpix_agg = valpix_agg / norm_3d
        
                rad_new = block_reduce(rad_new,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_new = rad_new / norm_3d
                rad_err_new = block_reduce(rad_err_new,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_err_new = rad_err_new / norm_3d 
                rad_err_new = rad_err_new / np.sqrt(xtrk_aggfac*atrk_aggfac)
                rad_new[valpix_agg <0.99999999999999999999] = np.nan
                rad_err_new[valpix_agg <0.99999999999999999999] = np.nan
        
                wvl_new = block_reduce(wvl_new,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; wvl_new = wvl_new / norm_3d
                rad_flags_new = block_reduce(rad_flags_new,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_flags_new = rad_flags_new / norm_3d
        
        
                rad_new = rad_new.transpose((2,1,0))
                rad_err_new = rad_err_new.transpose((2,1,0))
                wvl_new = wvl_new.transpose((2,1,0))
                rad_flags_new = rad_flags_new.transpose((2,1,0))
        
                ac_pos_new=ac_pos_new.transpose((1,0))
                ac_pix_bore_new=ac_pix_bore_new.transpose((2,1,0))
                l1 = pysplat.level1_AIR(l1_outfile,lon_new,lat_new,obsalt_new,time_new,ac_lon_new,ac_lat_new,ac_pos_new,ac_surf_alt_new,ac_alt_surf_new,ac_pix_bore_new,optbenchT=None,clon=corner_lon_new,clat=corner_lat_new,akaze_msi_image=akaze_msi_new)
                l1.set_2d_geofield('SurfaceAltitude', surfalt_new)
                l1.set_2d_geofield('SolarZenithAngle', sza_new)
                l1.set_2d_geofield('SolarAzimuthAngle', saa_new)
                l1.set_2d_geofield('ViewingZenithAngle', vza_new)
                l1.set_2d_geofield('ViewingAzimuthAngle', vaa_new)
                l1.set_2d_geofield('RelativeAzimuthAngle', aza_new)
                l1.add_radiance_band(wvl_new,rad_new,rad_err=rad_err_new,rad_flag=rad_flags_new)

                if(optimized_used_new==1):

                    l1.set_supportfield('AkazeLongitude',akz_lon_new)
                    l1.set_supportfield('AkazeLatitude',akz_lat_new)
                    l1.set_supportfield('AkazeSurfaceAltitude',akaze_surfalt_new)
                    l1.set_supportfield('AkazeCornerLatitude',akaze_clat_new)
                    l1.set_supportfield('AkazeCornerLongitude',akaze_clon_new)

                    l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt_new)
                    l1.set_supportfield('AvionicsSolarZenithAngle',av_sza_new)
                    l1.set_supportfield('AvionicsSolarAzimuthAngle',av_saa_new)
                    l1.set_supportfield('AvionicsViewingZenithAngle',av_vza_new)
                    l1.set_supportfield('AvionicsViewingAzimuthAngle',av_vaa_new)
                    l1.set_supportfield('AvionicsRelativeAzimuthAngle',av_aza_new)
                    l1.set_supportfield('AvionicsAircraftLongitude',av_ac_lon_new)
                    l1.set_supportfield('AvionicsAircraftLatitude',av_ac_lat_new)
                    l1.set_supportfield('AvionicsAircraftAltitudeAboveSurface',av_ac_alt_surf_new)
                    l1.set_supportfield('AvionicsAircraftSurfaceAltitude',av_ac_surf_alt_new)
                    l1.set_supportfield('AvionicsAircraftPixelBore',av_ac_pix_bore_new)
                    l1.set_supportfield('AvionicsAircraftPos',av_ac_pos_new)
                    l1.set_supportfield('AvionicsLongitude',av_lon_new)
                    l1.set_supportfield('AvionicsLatitude',av_lat_new)
                    l1.set_supportfield('AvionicsCornerLongitude',av_clon_new)
                    l1.set_supportfield('AvionicsCornerLatitude',av_clat_new)
                    l1.set_supportfield('AvionicsObservationAltitude',av_obsalt_new)
                    l1.set_1d_flag(True,None,None)
                    l1.add_akaze(ac_roll_new,ac_pitch_new,ac_heading_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new)

                elif(akaze_used_new==1):
                    l1.set_supportfield('OptimizedSolarZenithAngle', op_sza_new)
                    l1.set_supportfield('OptimizedSolarAzimuthAngle', op_saa_new)
                    l1.set_supportfield('OptimizedViewingZenithAngle', op_vza_new)
                    l1.set_supportfield('OptimizedViewingAzimuthAngle', op_vaa_new)
                    l1.set_supportfield('OptimizedRelativeAzimuthAngle', op_aza_new)
                    l1.set_supportfield('OptimizedSurfaceAltitude',op_surfalt_new)
                    l1.set_supportfield('OptimizedAircraftLongitude',op_ac_lon_new)
                    l1.set_supportfield('OptimizedAircraftLatitude',op_ac_lat_new)
                    l1.set_supportfield('OptimizedAircraftAltitudeAboveSurface',op_ac_alt_surf_new)
                    l1.set_supportfield('OptimizedAircraftSurfaceAltitude',op_ac_surf_alt_new)
                    l1.set_supportfield('OptimizedAircraftPixelBore',op_ac_pix_bore_new)
                    l1.set_supportfield('OptimizedAircraftPos',op_ac_pos_new)
                    l1.set_supportfield('OptimizedLongitude',op_lon_new)
                    l1.set_supportfield('OptimizedLatitude',op_lat_new)
                    l1.set_supportfield('OptimizedCornerLongitude',op_clon_new)
                    l1.set_supportfield('OptimizedCornerLatitude',op_clat_new)
                    l1.set_supportfield('OptimizedObservationAltitude',op_obsalt_new)

                    l1.set_supportfield('AvionicsLongitude',av_lon_new)
                    l1.set_supportfield('AvionicsLatitude',av_lat_new)
                    l1.set_supportfield('AvionicsCornerLongitude',av_clon_new)
                    l1.set_supportfield('AvionicsCornerLatitude',av_clat_new)
                    l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt_new)

                    l1.set_1d_flag(None,True,True)
                    l1.add_akaze(ac_roll_new,ac_pitch_new,ac_heading_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new)
                else:
                    l1.set_1d_flag(None,None,True)

               
                l1.close()
                ncfile = nc4.Dataset(l1_outfile,'a',format="NETCDF4")
                ncgroup = ncfile.createGroup('AlignmentMetaData') 
                data = ncgroup.createVariable('NativeXtrackShift',np.int16,('o'))
                data[:] = total_shift_in_pixels
                ncfile.close()
                found_match=False
                for abc in range(len(self.native_ch4_files)):
                    if(found_match==False and (str(os.path.basename(filech4) == str(self.native_ch4_files[abc]))) ):
                        priority =self.native_ch4_priority[abc]      
                        found_match=True
        
                lockname=logfile_ch4+'.lock'
                with FileLock(lockname):
                    f = open(logfile_ch4,'a+') 
                    f.write(str(l1_outfile)+' '+str(priority)+'\n' )
                    f.close()
        
        
        
                #############################
                # NOW WE WRITE THE NEW FILE TO DESK: CH4 FIRST
                #############################
                xtrk_aggfac = 3
                atrk_aggfac = 3
                filename = filech4.split(".nc")[0]
                filename = filename.split("CH4_NATIVE/")[1]
                l1b_ch4_dir = os.path.join(self.dest,'CH4_15x3_Aligned/') 
                l1_outfile = os.path.join(l1b_ch4_dir+filename+'.nc')
                logfile_ch4 = os.path.join(l1b_ch4_dir+'log_file_ch4_aligned.txt') 
                #l1_outfile = str(filename)+'.nc'
               
                # Flip all the data back to nornmal for 15x3.
                lon_new = lon_new.transpose((1,0))
                lat_new = lat_new.transpose((1,0))
                saa_new=saa_new.transpose((1,0))
                sza_new=sza_new.transpose((1,0))
                vza_new=vza_new.transpose((1,0))
                vaa_new=vaa_new.transpose((1,0))
                aza_new=aza_new.transpose((1,0))
                surfalt_new=surfalt_new.transpose((1,0))
                corner_lat_new = corner_lat_new.transpose((2,1,0))
                corner_lon_new = corner_lon_new.transpose((2,1,0)) 
                rad_new = rad_new.transpose((2,1,0))
                rad_err_new = rad_err_new.transpose((2,1,0))
                wvl_new = wvl_new.transpose((2,1,0))
                rad_flags_new = rad_flags_new.transpose((2,1,0))
        
                ac_pos_new=ac_pos_new.transpose((1,0))
                ac_pix_bore_new=ac_pix_bore_new.transpose((2,1,0))


                if(optimized_used_new==1):
                    av_lon_new = av_lon_new.transpose(1,0) 
                    av_lat_new = av_lat_new.transpose(1,0) 
                    av_sza_new = av_sza_new.transpose(1,0) 
                    av_saa_new = av_saa_new.transpose(1,0) 
                    av_aza_new = av_aza_new.transpose(1,0) 
                    av_vza_new = av_vza_new.transpose(1,0) 
                    av_vaa_new = av_vaa_new.transpose(1,0) 
                    av_surfalt_new = av_surfalt_new.transpose(1,0) 
                    av_clon_new = av_clon_new.transpose((2,1,0))
                    av_clat_new = av_clat_new.transpose((2,1,0))
                    akz_lon_new = akz_lon_new.transpose(1,0)
                    akz_lat_new = akz_lat_new.transpose(1,0)
                    akaze_msi_new = akaze_msi_new.transpose(1,0)
                    akaze_surfalt_new = akaze_surfalt_new.transpose(1,0)
                    akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new.transpose(1,0)
                    akaze_clon_new = akaze_clon_new.transpose((2,1,0))
                    akaze_clat_new = akaze_clat_new.transpose((2,1,0))
                    av_ac_pos_new=av_ac_pos_new.transpose((1,0))
                    av_ac_pix_bore_new=av_ac_pix_bore_new.transpose((2,1,0))

                if(akaze_used_new==1):
                    av_lon_new = av_lon_new.transpose(1,0) 
                    av_lat_new = av_lat_new.transpose(1,0) 
                    av_surfalt_new = av_surfalt_new.transpose(1,0) 
                    op_lon_new = op_lon_new.transpose(1,0) 
                    op_lat_new = op_lat_new.transpose(1,0) 
                    op_sza_new = op_sza_new.transpose(1,0) 
                    op_saa_new = op_saa_new.transpose(1,0) 
                    op_aza_new = op_aza_new.transpose(1,0) 
                    op_vza_new = op_vza_new.transpose(1,0) 
                    op_vaa_new = op_vaa_new.transpose(1,0) 
                    op_surfalt_new = op_surfalt_new.transpose(1,0) 
                    akaze_msi_new = akaze_msi_new.transpose(1,0)
                    akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new.transpose(1,0)
                    op_clon_new = op_clon_new.transpose((2,1,0))
                    op_clat_new = op_clat_new.transpose((2,1,0))
                    av_clon_new = av_clon_new.transpose((2,1,0))
                    av_clat_new = av_clat_new.transpose((2,1,0))
                    op_ac_pos_new=op_ac_pos_new.transpose((1,0))
                    op_ac_pix_bore_new=op_ac_pix_bore_new.transpose((2,1,0))


 
                norm_1d = block_reduce(np.ones(obsalt_new.shape),block_size=(atrk_aggfac,),func=np.mean)
                obsalt_new = block_reduce(obsalt_new,block_size=(atrk_aggfac,),func=np.mean) ; obsalt_new = obsalt_new / norm_1d
                time_new = block_reduce(time_new,block_size=(atrk_aggfac,),func=np.mean) ; time_new = time_new / norm_1d
                ac_alt_surf_new = block_reduce(ac_alt_surf_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_alt_surf_new = ac_alt_surf_new / norm_1d
                ac_surf_alt_new = block_reduce(ac_surf_alt_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_surf_alt_new = ac_surf_alt_new / norm_1d
                ac_lon_new= block_reduce(ac_lon_new,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lon_new = ac_lon_new / norm_1d
                ac_lat_new= block_reduce(ac_lat_new,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lat_new = ac_lat_new / norm_1d
                if(akaze_used_new==1):
                   ac_roll_new    = block_reduce(ac_roll_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll_new = ac_roll_new / norm_1d 
                   ac_pitch_new   = block_reduce(ac_pitch_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch_new = ac_pitch_new / norm_1d 
                   ac_heading_new = block_reduce(ac_heading_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading_new = ac_heading_new / norm_1d 
                   op_ac_lon_new  = block_reduce(op_ac_lon_new,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lon_new = op_ac_lon_new / norm_1d
                   op_ac_lat_new  = block_reduce(op_ac_lat_new,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lat_new = op_ac_lat_new / norm_1d
                   op_ac_alt_surf_new  = block_reduce(op_ac_alt_surf_new,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_alt_surf_new = op_ac_alt_surf_new/ norm_1d
                   op_ac_surf_alt_new  = block_reduce(op_ac_surf_alt_new,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_surf_alt_new = op_ac_surf_alt_new/ norm_1d
                   op_obsalt_new  = block_reduce(op_obsalt_new,block_size=(atrk_aggfac,),func=np.mean) ; op_obsalt_new = op_obsalt_new / norm_1d
                elif(optimized_used_new==1):
                   ac_roll_new    = block_reduce(ac_roll_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll_new = ac_roll_new / norm_1d 
                   ac_pitch_new   = block_reduce(ac_pitch_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch_new = ac_pitch_new / norm_1d 
                   ac_heading_new = block_reduce(ac_heading_new,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading_new = ac_heading_new / norm_1d 
                   av_ac_lon_new = block_reduce(av_ac_lon_new,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lon_new  =av_ac_lon_new/ norm_1d
                   av_ac_lat_new = block_reduce(av_ac_lat_new,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lat_new  =av_ac_lat_new/ norm_1d
                   av_ac_alt_surf_new = block_reduce(av_ac_alt_surf_new,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_alt_surf_new=av_ac_alt_surf_new    / norm_1d
                   av_ac_surf_alt_new = block_reduce(av_ac_surf_alt_new,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_surf_alt_new=av_ac_surf_alt_new    / norm_1d
                   av_obsalt_new = block_reduce(av_obsalt_new,block_size=(atrk_aggfac,),func=np.mean) ;av_obsalt_new =av_obsalt_new    / norm_1d
        
                norm_2d = block_reduce(np.ones(lon_new.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
                lon_new = block_reduce(lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lon_new = lon_new / norm_2d
                lat_new = block_reduce(lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lat_new = lat_new / norm_2d
                saa_new=block_reduce(saa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  saa_new = saa_new / norm_2d
                sza_new=block_reduce(sza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  sza_new = sza_new / norm_2d
                vza_new=block_reduce(vza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vza_new = vza_new / norm_2d
                vaa_new=block_reduce(vaa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vaa_new = vaa_new / norm_2d
                aza_new=block_reduce(aza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  aza_new = aza_new / norm_2d
                surfalt_new=block_reduce(surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  surfalt_new = surfalt_new / norm_2d
        
                lon_new = lon_new.transpose((1,0))
                lat_new = lat_new.transpose((1,0))
                saa_new=saa_new.transpose((1,0))
                sza_new=sza_new.transpose((1,0))
                vza_new=vza_new.transpose((1,0))
                vaa_new=vaa_new.transpose((1,0))
                aza_new=aza_new.transpose((1,0))
                surfalt_new=surfalt_new.transpose((1,0))
        
                if(optimized_used_new==1):
                    akz_lon_new = block_reduce(akz_lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akz_lon_new = akz_lon_new / norm_2d
                    akz_lat_new = block_reduce(akz_lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akz_lat_new = akz_lat_new / norm_2d
                    akaze_msi_new = block_reduce(akaze_msi_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi_new = akaze_msi_new / norm_2d
                    akaze_surfalt_new = block_reduce(akaze_surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_surfalt_new = akaze_surfalt_new / norm_2d 
                    akaze_Distance_Akaze_Reprojected_new = block_reduce(akaze_Distance_Akaze_Reprojected_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new / norm_2d

                    av_lon_new = block_reduce(av_lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon_new=av_lon_new   / norm_2d
                    av_lat_new = block_reduce(av_lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat_new=av_lat_new   / norm_2d
                    av_sza_new = block_reduce(av_sza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_sza_new=av_sza_new   / norm_2d
                    av_saa_new = block_reduce(av_saa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_saa_new=av_saa_new   / norm_2d
                    av_aza_new = block_reduce(av_aza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_aza_new=av_aza_new   / norm_2d
                    av_vza_new = block_reduce(av_vza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vza_new=av_vza_new   / norm_2d
                    av_vaa_new = block_reduce(av_vaa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vaa_new=av_vaa_new   / norm_2d
                    av_surfalt_new = block_reduce(av_surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt_new=av_surfalt_new/norm_2d 
                    av_lon_new = av_lon_new.transpose(1,0) 
                    av_lat_new = av_lat_new.transpose(1,0) 
                    av_sza_new = av_sza_new.transpose(1,0) 
                    av_saa_new = av_saa_new.transpose(1,0) 
                    av_aza_new = av_aza_new.transpose(1,0) 
                    av_vza_new = av_vza_new.transpose(1,0) 
                    av_vaa_new = av_vaa_new.transpose(1,0) 
                    av_surfalt_new = av_surfalt_new.transpose(1,0) 


                    norm_3d = block_reduce(np.ones(corner_lat_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    akaze_clon_new = block_reduce(akaze_clon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clon_new = akaze_clon_new / norm_3d 
                    akaze_clat_new = block_reduce(akaze_clat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clat_new = akaze_clat_new / norm_3d 
                    av_clon_new = block_reduce(av_clon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon_new = av_clon_new / norm_3d 
                    av_clat_new = block_reduce(av_clat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat_new = av_clat_new / norm_3d 
                    av_clon_new = av_clon_new.transpose((2,1,0))
                    av_clat_new = av_clat_new.transpose((2,1,0))
                    

                    akz_lon_new = akz_lon_new.transpose(1,0)
                    akz_lat_new = akz_lat_new.transpose(1,0)
                    akaze_msi_new = akaze_msi_new.transpose(1,0)
                    akaze_surfalt_new = akaze_surfalt_new.transpose(1,0)
                    akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new.transpose(1,0)
                    akaze_clon_new = akaze_clon_new.transpose((2,1,0))
                    akaze_clat_new = akaze_clat_new.transpose((2,1,0))

                    norm_3d = block_reduce(np.ones(ac_pix_bore_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    av_ac_pix_bore_new = block_reduce(av_ac_pix_bore_new,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_ac_pix_bore_new = av_ac_pix_bore_new/norm_3d 

                    norm_2d = block_reduce(np.ones(ac_pos_new.shape),(1,atrk_aggfac),func=np.mean)
                    av_ac_pos_new = block_reduce(av_ac_pos_new,block_size=(1,atrk_aggfac),func=np.mean) ;  av_ac_pos_new = av_ac_pos_new / norm_2d 

                    av_ac_pos_new=av_ac_pos_new.transpose((1,0))
                    av_ac_pix_bore_new=av_ac_pix_bore_new.transpose((2,1,0))

                if(akaze_used_new==1):
                    akaze_msi_new = block_reduce(akaze_msi_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi_new = akaze_msi_new / norm_2d
                    akaze_Distance_Akaze_Reprojected_new = block_reduce(akaze_Distance_Akaze_Reprojected_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new / norm_2d
                    av_lon_new = block_reduce(av_lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon_new=av_lon_new   / norm_2d
                    av_lat_new = block_reduce(av_lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat_new=av_lat_new   / norm_2d
                    av_surfalt_new = block_reduce(av_surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt_new=av_surfalt_new/norm_2d 
                    av_lon_new = av_lon_new.transpose(1,0) 
                    av_lat_new = av_lat_new.transpose(1,0) 
                    av_surfalt_new = av_surfalt_new.transpose(1,0) 

                    op_lon_new = block_reduce(op_lon_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lon_new=op_lon_new  / norm_2d
                    op_lat_new = block_reduce(op_lat_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lat_new=op_lat_new  / norm_2d
                    op_sza_new = block_reduce(op_sza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_sza_new=op_sza_new  / norm_2d
                    op_saa_new = block_reduce(op_saa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_saa_new=op_saa_new  / norm_2d
                    op_aza_new = block_reduce(op_aza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_aza_new=op_aza_new  / norm_2d
                    op_vza_new = block_reduce(op_vza_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vza_new=op_vza_new  / norm_2d
                    op_vaa_new = block_reduce(op_vaa_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vaa_new=op_vaa_new  / norm_2d
                    op_surfalt_new = block_reduce(op_surfalt_new,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_surfalt_new =op_surfalt_new / norm_2d

                    op_lon_new = op_lon_new.transpose(1,0) 
                    op_lat_new = op_lat_new.transpose(1,0) 
                    op_sza_new = op_sza_new.transpose(1,0) 
                    op_saa_new = op_saa_new.transpose(1,0) 
                    op_aza_new = op_aza_new.transpose(1,0) 
                    op_vza_new = op_vza_new.transpose(1,0) 
                    op_vaa_new = op_vaa_new.transpose(1,0) 
                    op_surfalt_new = op_surfalt_new.transpose(1,0) 
                    akaze_msi_new = akaze_msi_new.transpose(1,0)
                    akaze_Distance_Akaze_Reprojected_new = akaze_Distance_Akaze_Reprojected_new.transpose(1,0)



                    norm_3d = block_reduce(np.ones(corner_lat_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)

                    av_clon_new = block_reduce(av_clon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon_new = av_clon_new / norm_3d 
                    av_clat_new = block_reduce(av_clat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat_new = av_clat_new / norm_3d 
                    op_clon_new = block_reduce(op_clon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clon_new = op_clon_new / norm_3d 
                    op_clat_new = block_reduce(op_clat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clat_new = op_clat_new / norm_3d 
                    op_clon_new = op_clon_new.transpose((2,1,0))
                    op_clat_new = op_clat_new.transpose((2,1,0))
                    av_clon_new = av_clon_new.transpose((2,1,0))
                    av_clat_new = av_clat_new.transpose((2,1,0))

                    norm_3d = block_reduce(np.ones(ac_pix_bore_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    op_ac_pix_bore_new = block_reduce(op_ac_pix_bore_new,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_ac_pix_bore_new = op_ac_pix_bore_new/norm_3d 

                    norm_2d = block_reduce(np.ones(ac_pos_new.shape),(1,atrk_aggfac),func=np.mean)
                    op_ac_pos_new = block_reduce(op_ac_pos_new,block_size=(1,atrk_aggfac),func=np.mean) ;  op_ac_pos_new = op_ac_pos_new / norm_2d 

                    op_ac_pos_new=op_ac_pos_new.transpose((1,0))
                    op_ac_pix_bore_new=op_ac_pix_bore_new.transpose((2,1,0))
        
                norm_3d = block_reduce(np.ones(ac_pix_bore_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                ac_pix_bore_new=block_reduce(ac_pix_bore_new,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ;  ac_pix_bore_new = ac_pix_bore_new / norm_3d
        
                norm_2d = block_reduce(np.ones(ac_pos_new.shape),(1,atrk_aggfac),func=np.mean)
                ac_pos_new=block_reduce(ac_pos_new,block_size=(1,atrk_aggfac),func=np.mean) ;  ac_pos_new = ac_pos_new / norm_2d
        
        
                norm_3d = block_reduce(np.ones(corner_lat_new.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                corner_lat_new = block_reduce(corner_lat_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lat_new = corner_lat_new / norm_3d
                corner_lon_new = block_reduce(corner_lon_new,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lon_new = corner_lon_new / norm_3d
        
                corner_lat_new = corner_lat_new.transpose((2,1,0))
                corner_lon_new = corner_lon_new.transpose((2,1,0)) 
        
                norm_3d = block_reduce(np.ones(rad_new.shape),block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
                valpix = np.zeros(rad_new.shape)
                idv = np.logical_and(np.isfinite(rad_new),rad_new>0.0)
                valpix[idv] = 1.0
                valpix_agg = block_reduce(valpix,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
                valpix_agg = valpix_agg / norm_3d
        
                rad_new = block_reduce(rad_new,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_new = rad_new / norm_3d
                rad_err_new = block_reduce(rad_err_new,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_err_new = rad_err_new / norm_3d 
                rad_err_new = rad_err_new / np.sqrt(xtrk_aggfac*atrk_aggfac)
                rad_new[valpix_agg <0.99999999999999999999] = np.nan
                rad_err_new[valpix_agg <0.99999999999999999999] = np.nan
        
                wvl_new = block_reduce(wvl_new,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; wvl_new = wvl_new / norm_3d
                rad_flags_new = block_reduce(rad_flags_new,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_flags_new = rad_flags_new / norm_3d
        
        
                rad_new = rad_new.transpose((2,1,0))
                rad_err_new = rad_err_new.transpose((2,1,0))
                wvl_new = wvl_new.transpose((2,1,0))
                rad_flags_new = rad_flags_new.transpose((2,1,0))
        
                ac_pos_new=ac_pos_new.transpose((1,0))
                ac_pix_bore_new=ac_pix_bore_new.transpose((2,1,0))

                l1 = pysplat.level1_AIR(l1_outfile,lon_new,lat_new,obsalt_new,time_new,ac_lon_new,ac_lat_new,ac_pos_new,ac_surf_alt_new,ac_alt_surf_new,ac_pix_bore_new,optbenchT=None,clon=corner_lon_new,clat=corner_lat_new,akaze_msi_image=akaze_msi_new)
                l1.set_2d_geofield('SurfaceAltitude', surfalt_new)
                l1.set_2d_geofield('SolarZenithAngle', sza_new)
                l1.set_2d_geofield('SolarAzimuthAngle', saa_new)
                l1.set_2d_geofield('ViewingZenithAngle', vza_new)
                l1.set_2d_geofield('ViewingAzimuthAngle', vaa_new)
                l1.set_2d_geofield('RelativeAzimuthAngle', aza_new)
                l1.add_radiance_band(wvl_new,rad_new,rad_err=rad_err_new,rad_flag=rad_flags_new)

                if(optimized_used_new==1):
                    l1.set_supportfield('AkazeLongitude',akz_lon_new)
                    l1.set_supportfield('AkazeLatitude',akz_lat_new)
                    l1.set_supportfield('AkazeSurfaceAltitude',akaze_surfalt_new)
                    l1.set_supportfield('AkazeCornerLatitude',akaze_clat_new)
                    l1.set_supportfield('AkazeCornerLongitude',akaze_clon_new)

                    l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt_new)
                    l1.set_supportfield('AvionicsSolarZenithAngle',av_sza_new)
                    l1.set_supportfield('AvionicsSolarAzimuthAngle',av_saa_new)
                    l1.set_supportfield('AvionicsViewingZenithAngle',av_vza_new)
                    l1.set_supportfield('AvionicsViewingAzimuthAngle',av_vaa_new)
                    l1.set_supportfield('AvionicsRelativeAzimuthAngle',av_aza_new)
                    l1.set_supportfield('AvionicsAircraftLongitude',av_ac_lon_new)
                    l1.set_supportfield('AvionicsAircraftLatitude',av_ac_lat_new)
                    l1.set_supportfield('AvionicsAircraftAltitudeAboveSurface',av_ac_alt_surf_new)
                    l1.set_supportfield('AvionicsAircraftSurfaceAltitude',av_ac_surf_alt_new)
                    l1.set_supportfield('AvionicsAircraftPixelBore',av_ac_pix_bore_new)
                    l1.set_supportfield('AvionicsAircraftPos',av_ac_pos_new)
                    l1.set_supportfield('AvionicsLongitude',av_lon_new)
                    l1.set_supportfield('AvionicsLatitude',av_lat_new)
                    l1.set_supportfield('AvionicsCornerLongitude',av_clon_new)
                    l1.set_supportfield('AvionicsCornerLatitude',av_clat_new)
                    l1.set_supportfield('AvionicsObservationAltitude',av_obsalt_new)
                    l1.set_1d_flag(True,None,None)
                    l1.add_akaze(ac_roll_new,ac_pitch_new,ac_heading_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new)

                elif(akaze_used_new==1):
                    l1.set_supportfield('OptimizedSolarZenithAngle', op_sza_new)
                    l1.set_supportfield('OptimizedSolarAzimuthAngle', op_saa_new)
                    l1.set_supportfield('OptimizedViewingZenithAngle', op_vza_new)
                    l1.set_supportfield('OptimizedViewingAzimuthAngle', op_vaa_new)
                    l1.set_supportfield('OptimizedRelativeAzimuthAngle', op_aza_new)
                    l1.set_supportfield('OptimizedSurfaceAltitude',op_surfalt_new)
                    l1.set_supportfield('OptimizedAircraftLongitude',op_ac_lon_new)
                    l1.set_supportfield('OptimizedAircraftLatitude',op_ac_lat_new)
                    l1.set_supportfield('OptimizedAircraftAltitudeAboveSurface',op_ac_alt_surf_new)
                    l1.set_supportfield('OptimizedAircraftSurfaceAltitude',op_ac_surf_alt_new)
                    l1.set_supportfield('OptimizedAircraftPixelBore',op_ac_pix_bore_new)
                    l1.set_supportfield('OptimizedAircraftPos',op_ac_pos_new)
                    l1.set_supportfield('OptimizedLongitude',op_lon_new)
                    l1.set_supportfield('OptimizedLatitude',op_lat_new)
                    l1.set_supportfield('OptimizedCornerLongitude',op_clon_new)
                    l1.set_supportfield('OptimizedCornerLatitude',op_clat_new)
                    l1.set_supportfield('OptimizedObservationAltitude',op_obsalt_new)

                    l1.set_supportfield('AvionicsLongitude',av_lon_new)
                    l1.set_supportfield('AvionicsLatitude',av_lat_new)
                    l1.set_supportfield('AvionicsCornerLongitude',av_clon_new)
                    l1.set_supportfield('AvionicsCornerLatitude',av_clat_new)
                    l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt_new)

                    l1.set_1d_flag(None,True,True)
                    l1.add_akaze(ac_roll_new,ac_pitch_new,ac_heading_new,akaze_Distance_Akaze_Reprojected_new,akaze_Optimization_Convergence_Fail_new,akaze_Reprojection_Fit_Flag_new)
                else:
                    l1.set_1d_flag(None,None,True)


                l1.close()
                ncfile = nc4.Dataset(l1_outfile,'a',format="NETCDF4")
                ncgroup = ncfile.createGroup('AlignmentMetaData') 
                data = ncgroup.createVariable('NativeXtrackShift',np.int16,('o'))
                data[:] = total_shift_in_pixels
                ncfile.close()
                
                  
                found_match=False
                for abc in range(len(self.native_ch4_files)):
                    if(found_match==False and (str(os.path.basename(filech4) == str(self.native_ch4_files[abc]))) ):
                        priority =self.native_ch4_priority[abc]      
                        found_match=True
        
                lockname=logfile_ch4+'.lock'
                with FileLock(lockname):
                    f = open(logfile_ch4,'a+') 
                    f.write(str(l1_outfile)+' '+str(priority)+'\n' )
                    f.close()
        
        
                if(fileo2 is not None):
                    #############################
                    # NOW WE WRITE THE NEW FILE TO DESK: O2 SECOND
                    #############################
                    xtrk_aggfac = 5
                    atrk_aggfac = 1
        
                    filename = fileo2.split(".nc")[0]
                    filename = filename.split("O2_NATIVE/")[1]
                    l1b_o2_dir = os.path.join(self.dest,'O2_5x1_Aligned/') 
                    l1_outfile = os.path.join(l1b_o2_dir+filename+'.nc')
                    logfile_o2 = os.path.join(l1b_o2_dir+'log_file_o2_aligned.txt') 
                    #l1_outfile = str(filename)+'.nc'
        
                    norm_1d = block_reduce(np.ones(obsalt_o2.shape),block_size=(atrk_aggfac,),func=np.mean)
                    obsalt_o2 = block_reduce(obsalt_o2,block_size=(atrk_aggfac,),func=np.mean) ; obsalt_o2 = obsalt_o2 / norm_1d
                    time_o2 = block_reduce(time_o2,block_size=(atrk_aggfac,),func=np.mean) ; time_o2 = time_o2 / norm_1d
                    ac_alt_surf_o2 = block_reduce(ac_alt_surf_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_alt_surf_o2 = ac_alt_surf_o2 / norm_1d
                    ac_surf_alt_o2 = block_reduce(ac_surf_alt_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_surf_alt_o2 = ac_surf_alt_o2 / norm_1d
                    ac_lon_o2= block_reduce(ac_lon_o2,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lon_o2 = ac_lon_o2 / norm_1d
                    ac_lat_o2= block_reduce(ac_lat_o2,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lat_o2 = ac_lat_o2 / norm_1d
                    if(akaze_used==1):
                       ac_roll_o2    = block_reduce(ac_roll_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll_o2 = ac_roll_o2 / norm_1d 
                       ac_pitch_o2   = block_reduce(ac_pitch_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch_o2 = ac_pitch_o2 / norm_1d 
                       ac_heading_o2 = block_reduce(ac_heading_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading_o2 = ac_heading_o2 / norm_1d 
                       op_ac_lon_o2  = block_reduce(op_ac_lon_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lon_o2 = op_ac_lon_o2 / norm_1d
                       op_ac_lat_o2  = block_reduce(op_ac_lat_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lat_o2 = op_ac_lat_o2 / norm_1d
                       op_ac_alt_surf_o2  = block_reduce(op_ac_alt_surf_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_alt_surf_o2 = op_ac_alt_surf_o2/ norm_1d
                       op_ac_surf_alt_o2  = block_reduce(op_ac_surf_alt_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_surf_alt_o2 = op_ac_surf_alt_o2/ norm_1d
                       op_obsalt_o2  = block_reduce(op_obsalt_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_obsalt_o2 = op_obsalt_o2 / norm_1d
                    elif(optimized_used==1):
                       ac_roll_o2    = block_reduce(ac_roll_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll_o2 = ac_roll_o2 / norm_1d 
                       ac_pitch_o2   = block_reduce(ac_pitch_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch_o2 = ac_pitch_o2 / norm_1d 
                       ac_heading_o2 = block_reduce(ac_heading_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading_o2 = ac_heading_o2 / norm_1d 
                       av_ac_lon_o2 = block_reduce(av_ac_lon_o2,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lon_o2  =av_ac_lon_o2/ norm_1d
                       av_ac_lat_o2 = block_reduce(av_ac_lat_o2,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lat_o2  =av_ac_lat_o2/ norm_1d
                       av_ac_alt_surf_o2 = block_reduce(av_ac_alt_surf_o2,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_alt_surf_o2=av_ac_alt_surf_o2    / norm_1d
                       av_ac_surf_alt_o2 = block_reduce(av_ac_surf_alt_o2,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_surf_alt_o2=av_ac_surf_alt_o2    / norm_1d
                       av_obsalt_o2 = block_reduce(av_obsalt_o2,block_size=(atrk_aggfac,),func=np.mean) ;av_obsalt_o2 =av_obsalt_o2    / norm_1d
        
                    norm_2d = block_reduce(np.ones(lon_o2.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
                    lon_o2 = block_reduce(lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lon_o2 = lon_o2 / norm_2d
                    lat_o2 = block_reduce(lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lat_o2 = lat_o2 / norm_2d
                    saa_o2=block_reduce(saa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  saa_o2 = saa_o2 / norm_2d
                    sza_o2=block_reduce(sza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  sza_o2 = sza_o2 / norm_2d
                    vza_o2=block_reduce(vza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vza_o2 = vza_o2 / norm_2d
                    vaa_o2=block_reduce(vaa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vaa_o2 = vaa_o2 / norm_2d
                    aza_o2=block_reduce(aza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  aza_o2 = aza_o2 / norm_2d
                    surfalt_o2=block_reduce(surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  surfalt_o2 = surfalt_o2 / norm_2d

                    if(akaze_used==1):
                        akaze_msi_o2 = block_reduce(akaze_msi_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi_o2 = akaze_msi_o2 / norm_2d
                        akaze_Distance_Akaze_Reprojected_o2 = block_reduce(akaze_Distance_Akaze_Reprojected_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2 / norm_2d
                        av_lon_o2 = block_reduce(av_lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon_o2=av_lon_o2   / norm_2d
                        av_lat_o2 = block_reduce(av_lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat_o2=av_lat_o2   / norm_2d
                        av_surfalt_o2 = block_reduce(av_surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt_o2=av_surfalt_o2/norm_2d 
                        av_lon_o2 = av_lon_o2.transpose(1,0) 
                        av_lat_o2 = av_lat_o2.transpose(1,0) 
                        av_surfalt_o2 = av_surfalt_o2.transpose(1,0) 

                        op_lon_o2 = block_reduce(op_lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lon_o2=op_lon_o2  / norm_2d
                        op_lat_o2 = block_reduce(op_lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lat_o2=op_lat_o2  / norm_2d
                        op_sza_o2 = block_reduce(op_sza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_sza_o2=op_sza_o2  / norm_2d
                        op_saa_o2 = block_reduce(op_saa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_saa_o2=op_saa_o2  / norm_2d
                        op_aza_o2 = block_reduce(op_aza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_aza_o2=op_aza_o2  / norm_2d
                        op_vza_o2 = block_reduce(op_vza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vza_o2=op_vza_o2  / norm_2d
                        op_vaa_o2 = block_reduce(op_vaa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vaa_o2=op_vaa_o2  / norm_2d
                        op_surfalt_o2 = block_reduce(op_surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_surfalt_o2 =op_surfalt_o2 / norm_2d

                        op_lon_o2 = op_lon_o2.transpose(1,0) 
                        op_lat_o2 = op_lat_o2.transpose(1,0) 
                        op_sza_o2 = op_sza_o2.transpose(1,0) 
                        op_saa_o2 = op_saa_o2.transpose(1,0) 
                        op_aza_o2 = op_aza_o2.transpose(1,0) 
                        op_vza_o2 = op_vza_o2.transpose(1,0) 
                        op_vaa_o2 = op_vaa_o2.transpose(1,0) 
                        op_surfalt_o2 = op_surfalt_o2.transpose(1,0) 
                        akaze_msi_o2 = akaze_msi_o2.transpose(1,0)
                        akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2.transpose(1,0)


                        norm_3d = block_reduce(np.ones(corner_lat_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)

                        av_clon_o2 = block_reduce(av_clon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon_o2 = av_clon_o2 / norm_3d 
                        av_clat_o2 = block_reduce(av_clat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat_o2 = av_clat_o2 / norm_3d 
                        op_clon_o2 = block_reduce(op_clon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clon_o2 = op_clon_o2 / norm_3d 
                        op_clat_o2 = block_reduce(op_clat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clat_o2 = op_clat_o2 / norm_3d 
                        op_clon_o2 = op_clon_o2.transpose((2,1,0))
                        op_clat_o2 = op_clat_o2.transpose((2,1,0))
                        av_clon_o2 = av_clon_o2.transpose((2,1,0))
                        av_clat_o2 = av_clat_o2.transpose((2,1,0))

                        norm_3d = block_reduce(np.ones(ac_pix_bore_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                        op_ac_pix_bore_o2 = block_reduce(op_ac_pix_bore_o2,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_ac_pix_bore_o2 = op_ac_pix_bore_o2/norm_3d 

                        norm_2d = block_reduce(np.ones(ac_pos_o2.shape),(1,atrk_aggfac),func=np.mean)
                        op_ac_pos_o2 = block_reduce(op_ac_pos_o2,block_size=(1,atrk_aggfac),func=np.mean) ;  op_ac_pos_o2 = op_ac_pos_o2 / norm_2d 

                        op_ac_pos_o2=op_ac_pos_o2.transpose((1,0))
                        op_ac_pix_bore_o2=op_ac_pix_bore_o2.transpose((2,1,0))

                    elif(optimized_used==1):
                        akaze_lon_o2 = block_reduce(akaze_lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_lon_o2 = akaze_lon_o2 / norm_2d
                        akaze_lat_o2 = block_reduce(akaze_lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_lat_o2 = akaze_lat_o2 / norm_2d
                        akaze_msi_o2 = block_reduce(akaze_msi_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi_o2 = akaze_msi_o2 / norm_2d
                        akaze_surfalt_o2 = block_reduce(akaze_surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_surfalt_o2 = akaze_surfalt_o2 / norm_2d 
                        akaze_Distance_Akaze_Reprojected_o2 = block_reduce(akaze_Distance_Akaze_Reprojected_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2 / norm_2d

                        av_lon_o2 = block_reduce(av_lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon_o2=av_lon_o2   / norm_2d
                        av_lat_o2 = block_reduce(av_lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat_o2=av_lat_o2   / norm_2d
                        av_sza_o2 = block_reduce(av_sza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_sza_o2=av_sza_o2   / norm_2d
                        av_saa_o2 = block_reduce(av_saa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_saa_o2=av_saa_o2   / norm_2d
                        av_aza_o2 = block_reduce(av_aza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_aza_o2=av_aza_o2   / norm_2d
                        av_vza_o2 = block_reduce(av_vza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vza_o2=av_vza_o2   / norm_2d
                        av_vaa_o2 = block_reduce(av_vaa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vaa_o2=av_vaa_o2   / norm_2d
                        av_surfalt_o2 = block_reduce(av_surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt_o2=av_surfalt_o2/norm_2d 
                        av_lon_o2 = av_lon_o2.transpose(1,0) 
                        av_lat_o2 = av_lat_o2.transpose(1,0) 
                        av_sza_o2 = av_sza_o2.transpose(1,0) 
                        av_saa_o2 = av_saa_o2.transpose(1,0) 
                        av_aza_o2 = av_aza_o2.transpose(1,0) 
                        av_vza_o2 = av_vza_o2.transpose(1,0) 
                        av_vaa_o2 = av_vaa_o2.transpose(1,0) 
                        av_surfalt_o2 = av_surfalt_o2.transpose(1,0) 


                        norm_3d = block_reduce(np.ones(corner_lat_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                        akaze_clon_o2 = block_reduce(akaze_clon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clon_o2 = akaze_clon_o2 / norm_3d 
                        akaze_clat_o2 = block_reduce(akaze_clat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clat_o2 = akaze_clat_o2 / norm_3d 
                        av_clon_o2 = block_reduce(av_clon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon_o2 = av_clon_o2 / norm_3d 
                        av_clat_o2 = block_reduce(av_clat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat_o2 = av_clat_o2 / norm_3d 
                        av_clon_o2 = av_clon_o2.transpose((2,1,0))
                        av_clat_o2 = av_clat_o2.transpose((2,1,0))
                        

                        akaze_lon_o2 = akaze_lon_o2.transpose(1,0)
                        akaze_lat_o2 = akaze_lat_o2.transpose(1,0)
                        akaze_msi_o2 = akaze_msi_o2.transpose(1,0)
                        akaze_surfalt_o2 = akaze_surfalt_o2.transpose(1,0)
                        akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2.transpose(1,0)
                        akaze_clon_o2 = akaze_clon_o2.transpose((2,1,0))
                        akaze_clat_o2 = akaze_clat_o2.transpose((2,1,0))

                        norm_3d = block_reduce(np.ones(ac_pix_bore_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                        av_ac_pix_bore_o2 = block_reduce(av_ac_pix_bore_o2,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_ac_pix_bore_o2 = av_ac_pix_bore_o2/norm_3d 

                        norm_2d = block_reduce(np.ones(ac_pos_o2.shape),(1,atrk_aggfac),func=np.mean)
                        av_ac_pos_o2 = block_reduce(av_ac_pos_o2,block_size=(1,atrk_aggfac),func=np.mean) ;  av_ac_pos_o2 = av_ac_pos_o2 / norm_2d 

                        av_ac_pos_o2=av_ac_pos_o2.transpose((1,0))
                        av_ac_pix_bore_o2=av_ac_pix_bore_o2.transpose((2,1,0))

        
                    lon_o2 = lon_o2.transpose((1,0))
                    lat_o2 = lat_o2.transpose((1,0))
                    saa_o2=saa_o2.transpose((1,0))
                    sza_o2=sza_o2.transpose((1,0))
                    vza_o2=vza_o2.transpose((1,0))
                    vaa_o2=vaa_o2.transpose((1,0))
                    aza_o2=aza_o2.transpose((1,0))
                    surfalt_o2=surfalt_o2.transpose((1,0))
                    #ac_pos_o2=ac_pos_o2.transpose((1,0))
                    #ac_pix_bore_o2=ac_pix_bore_o2.transpose((2,1,0))
        
                    norm_3d = block_reduce(np.ones(ac_pix_bore_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    ac_pix_bore_o2=block_reduce(ac_pix_bore_o2,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ;  ac_pix_bore_o2 = ac_pix_bore_o2 / norm_3d
        
                    norm_2d = block_reduce(np.ones(ac_pos_o2.shape),(1,atrk_aggfac),func=np.mean)
                    ac_pos_o2=block_reduce(ac_pos_o2,block_size=(1,atrk_aggfac),func=np.mean) ;  ac_pos_o2 = ac_pos_o2 / norm_2d
        
                    norm_3d = block_reduce(np.ones(corner_lat_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    corner_lat_o2 = block_reduce(corner_lat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lat_o2 = corner_lat_o2 / norm_3d
                    corner_lon_o2 = block_reduce(corner_lon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lon_o2 = corner_lon_o2 / norm_3d
        
        
                    norm_3d = block_reduce(np.ones(rad_o2.shape),block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
                    valpix = np.zeros(rad_o2.shape)
                    idv = np.logical_and(np.isfinite(rad_o2),rad_o2>0.0)
                    valpix[idv] = 1.0
                    valpix_agg = block_reduce(valpix,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
                    valpix_agg = valpix_agg / norm_3d
        
                    rad_o2 = block_reduce(rad_o2,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_o2 = rad_o2 / norm_3d
                    rad_err_o2 = block_reduce(rad_err_o2,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_err_o2 = rad_err_o2 / norm_3d 
                    rad_err_o2 = rad_err_o2 / np.sqrt(xtrk_aggfac*atrk_aggfac)
                    rad_o2[valpix_agg <0.99999999999999999999] = np.nan
                    rad_err_o2[valpix_agg <0.99999999999999999999] = np.nan
        
                    wvl_o2 = block_reduce(wvl_o2,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; wvl_o2 = wvl_o2 / norm_3d
                    rad_flags_o2 = block_reduce(rad_flags_o2,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_flags_o2 = rad_flags_o2 / norm_3d
        
        
        
        
                    corner_lat_o2 = corner_lat_o2.transpose((2,1,0))
                    corner_lon_o2 = corner_lon_o2.transpose((2,1,0))
                    wvl_o2 = wvl_o2.transpose((2,1,0))
                    rad_o2 = rad_o2.transpose((2,1,0))
                    rad_err_o2 = rad_err_o2.transpose((2,1,0))
                    rad_flags_o2 = rad_flags_o2.transpose((2,1,0))
        
                    ac_pos_o2=ac_pos_o2.transpose((1,0))
                    ac_pix_bore_o2=ac_pix_bore_o2.transpose((2,1,0))
                    

                    l1 = pysplat.level1_AIR(l1_outfile,lon_o2,lat_o2,obsalt_o2,time_o2,ac_lon_o2,ac_lat_o2,ac_pos_o2,ac_surf_alt_o2,ac_alt_surf_o2,ac_pix_bore_o2,optbenchT=None,clon=corner_lon_o2,clat=corner_lat_o2,akaze_msi_image=akaze_msi_o2)
                    l1.set_2d_geofield('SurfaceAltitude', surfalt_o2)
                    l1.set_2d_geofield('SolarZenithAngle', sza_o2)
                    l1.set_2d_geofield('SolarAzimuthAngle', saa_o2)
                    l1.set_2d_geofield('ViewingZenithAngle', vza_o2)
                    l1.set_2d_geofield('ViewingAzimuthAngle', vaa_o2)
                    l1.set_2d_geofield('RelativeAzimuthAngle', aza_o2)
                    l1.add_radiance_band(wvl_o2,rad_o2,rad_err=rad_err_o2,rad_flag=rad_flags_o2)

                    if(optimized_used==1):

                        l1.set_supportfield('AkazeLongitude',akaze_lon_o2)
                        l1.set_supportfield('AkazeLatitude',akaze_lat_o2)
                        l1.set_supportfield('AkazeSurfaceAltitude',akaze_surfalt_o2)
                        l1.set_supportfield('AkazeCornerLatitude',akaze_clat_o2)
                        l1.set_supportfield('AkazeCornerLongitude',akaze_clon_o2)

                        l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt_o2)
                        l1.set_supportfield('AvionicsSolarZenithAngle',av_sza_o2)
                        l1.set_supportfield('AvionicsSolarAzimuthAngle',av_saa_o2)
                        l1.set_supportfield('AvionicsViewingZenithAngle',av_vza_o2)
                        l1.set_supportfield('AvionicsViewingAzimuthAngle',av_vaa_o2)
                        l1.set_supportfield('AvionicsRelativeAzimuthAngle',av_aza_o2)
                        l1.set_supportfield('AvionicsAircraftLongitude',av_ac_lon_o2)
                        l1.set_supportfield('AvionicsAircraftLatitude',av_ac_lat_o2)
                        l1.set_supportfield('AvionicsAircraftAltitudeAboveSurface',av_ac_alt_surf_o2)
                        l1.set_supportfield('AvionicsAircraftSurfaceAltitude',av_ac_surf_alt_o2)
                        l1.set_supportfield('AvionicsAircraftPixelBore',av_ac_pix_bore_o2)
                        l1.set_supportfield('AvionicsAircraftPos',av_ac_pos_o2)
                        l1.set_supportfield('AvionicsLongitude',av_lon_o2)
                        l1.set_supportfield('AvionicsLatitude',av_lat_o2)
                        l1.set_supportfield('AvionicsCornerLongitude',av_clon_o2)
                        l1.set_supportfield('AvionicsCornerLatitude',av_clat_o2)
                        l1.set_supportfield('AvionicsObservationAltitude',av_obsalt_o2)
                        l1.set_1d_flag(True,None,None)
                        l1.add_akaze(ac_roll_o2,ac_pitch_o2,ac_heading_o2,akaze_Distance_Akaze_Reprojected_o2,akaze_Optimization_Convergence_Fail_o2,akaze_Reprojection_Fit_Flag_o2)

                    elif(akaze_used==1):
                        l1.set_supportfield('OptimizedSolarZenithAngle', op_sza_o2)
                        l1.set_supportfield('OptimizedSolarAzimuthAngle', op_saa_o2)
                        l1.set_supportfield('OptimizedViewingZenithAngle', op_vza_o2)
                        l1.set_supportfield('OptimizedViewingAzimuthAngle', op_vaa_o2)
                        l1.set_supportfield('OptimizedRelativeAzimuthAngle', op_aza_o2)
                        l1.set_supportfield('OptimizedSurfaceAltitude',op_surfalt_o2)
                        l1.set_supportfield('OptimizedAircraftLongitude',op_ac_lon_o2)
                        l1.set_supportfield('OptimizedAircraftLatitude',op_ac_lat_o2)
                        l1.set_supportfield('OptimizedAircraftAltitudeAboveSurface',op_ac_alt_surf_o2)
                        l1.set_supportfield('OptimizedAircraftSurfaceAltitude',op_ac_surf_alt_o2)
                        l1.set_supportfield('OptimizedAircraftPixelBore',op_ac_pix_bore_o2)
                        l1.set_supportfield('OptimizedAircraftPos',op_ac_pos_o2)
                        l1.set_supportfield('OptimizedLongitude',op_lon_o2)
                        l1.set_supportfield('OptimizedLatitude',op_lat_o2)
                        l1.set_supportfield('OptimizedCornerLongitude',op_clon_o2)
                        l1.set_supportfield('OptimizedCornerLatitude',op_clat_o2)
                        l1.set_supportfield('OptimizedObservationAltitude',op_obsalt_o2)

                        l1.set_supportfield('AvionicsLongitude',av_lon_o2)
                        l1.set_supportfield('AvionicsLatitude',av_lat_o2)
                        l1.set_supportfield('AvionicsCornerLongitude',av_clon_o2)
                        l1.set_supportfield('AvionicsCornerLatitude',av_clat_o2)
                        l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt_o2)

                        l1.set_1d_flag(None,True,True)
                        l1.add_akaze(ac_roll_o2,ac_pitch_o2,ac_heading_o2,akaze_Distance_Akaze_Reprojected_o2,akaze_Optimization_Convergence_Fail_o2,akaze_Reprojection_Fit_Flag_o2)
                    else:
                        l1.set_1d_flag(None,None,True)


                    l1.close()
                    ncfile = nc4.Dataset(l1_outfile,'a',format="NETCDF4")
                    ncgroup = ncfile.createGroup('AlignmentMetaData') 
                    data = ncgroup.createVariable('NativeXtrackShift',np.int16,('o'))
                    data[:] = total_shift_in_pixels
                    ncfile.close()
        
                    lockname=logfile_o2+'.lock'
                    with FileLock(lockname):
                        f = open(logfile_o2,'a+') 
                        f.write(str(l1_outfile)+' '+str(priority)+'\n' )
                        f.close()
                      
                    #############################
                    # NOW WE WRITE THE NEW FILE TO DESK: O2 SECOND
                    #############################
        
                    filename = fileo2.split(".nc")[0]
                    filename = filename.split("O2_NATIVE/")[1]
                    l1b_o2_dir = os.path.join(self.dest,'O2_15x3_Aligned/') 
                    l1_outfile = os.path.join(l1b_o2_dir+filename+'.nc')
                    logfile_o2 = os.path.join(l1b_o2_dir+'log_file_o2_aligned.txt') 
                    #l1_outfile = str(filename)+'.nc'
                    atrk_aggfac=3 
                    xtrk_aggfac=3 
                    lon_o2 = lon_o2.transpose((1,0))
                    lat_o2 = lat_o2.transpose((1,0))
                    saa_o2=saa_o2.transpose((1,0))
                    sza_o2=sza_o2.transpose((1,0))
                    vza_o2=vza_o2.transpose((1,0))
                    vaa_o2=vaa_o2.transpose((1,0))
                    aza_o2=aza_o2.transpose((1,0))
                    surfalt_o2=surfalt_o2.transpose((1,0))
                    corner_lat_o2 = corner_lat_o2.transpose((2,1,0))
                    corner_lon_o2 = corner_lon_o2.transpose((2,1,0))
                    wvl_o2 = wvl_o2.transpose((2,1,0))
                    rad_o2 = rad_o2.transpose((2,1,0))
                    rad_err_o2 = rad_err_o2.transpose((2,1,0))
                    rad_flags_o2 = rad_flags_o2.transpose((2,1,0))
                    ac_pos_o2=ac_pos_o2.transpose((1,0))
                    ac_pix_bore_o2=ac_pix_bore_o2.transpose((2,1,0))


                    if(optimized_used==1):
                        av_lon_o2 = av_lon_o2.transpose(1,0) 
                        av_lat_o2 = av_lat_o2.transpose(1,0) 
                        av_sza_o2 = av_sza_o2.transpose(1,0) 
                        av_saa_o2 = av_saa_o2.transpose(1,0) 
                        av_aza_o2 = av_aza_o2.transpose(1,0) 
                        av_vza_o2 = av_vza_o2.transpose(1,0) 
                        av_vaa_o2 = av_vaa_o2.transpose(1,0) 
                        av_surfalt_o2 = av_surfalt_o2.transpose(1,0) 
                        akaze_lon_o2 = akaze_lon_o2.transpose(1,0)
                        akaze_lat_o2 = akaze_lat_o2.transpose(1,0)
                        akaze_msi_o2 = akaze_msi_o2.transpose(1,0)
                        akaze_surfalt_o2 = akaze_surfalt_o2.transpose(1,0)
                        akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2.transpose(1,0)
                        akaze_clon_o2 = akaze_clon_o2.transpose((2,1,0))
                        akaze_clat_o2 = akaze_clat_o2.transpose((2,1,0))
                        av_clon_o2 = av_clon_o2.transpose((2,1,0))
                        av_clat_o2 = av_clat_o2.transpose((2,1,0))
                        av_ac_pos_o2=av_ac_pos_o2.transpose((1,0))
                        av_ac_pix_bore_o2=av_ac_pix_bore_o2.transpose((2,1,0))

                    if(akaze_used==1):
                        av_lon_o2 = av_lon_o2.transpose(1,0) 
                        av_lat_o2 = av_lat_o2.transpose(1,0) 
                        av_surfalt_o2 = av_surfalt_o2.transpose(1,0) 
                        op_lon_o2 = op_lon_o2.transpose(1,0) 
                        op_lat_o2 = op_lat_o2.transpose(1,0) 
                        op_sza_o2 = op_sza_o2.transpose(1,0) 
                        op_saa_o2 = op_saa_o2.transpose(1,0) 
                        op_aza_o2 = op_aza_o2.transpose(1,0) 
                        op_vza_o2 = op_vza_o2.transpose(1,0) 
                        op_vaa_o2 = op_vaa_o2.transpose(1,0) 
                        op_surfalt_o2 = op_surfalt_o2.transpose(1,0) 
                        akaze_msi_o2 = akaze_msi_o2.transpose(1,0)
                        akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2.transpose(1,0)
                        op_clon_o2 = op_clon_o2.transpose((2,1,0))
                        op_clat_o2 = op_clat_o2.transpose((2,1,0))
                        av_clon_o2 = av_clon_o2.transpose((2,1,0))
                        av_clat_o2 = av_clat_o2.transpose((2,1,0))
                        op_ac_pos_o2=op_ac_pos_o2.transpose((1,0))
                        op_ac_pix_bore_o2=op_ac_pix_bore_o2.transpose((2,1,0))



                    norm_1d = block_reduce(np.ones(obsalt_o2.shape),block_size=(atrk_aggfac,),func=np.mean)
                    obsalt_o2 = block_reduce(obsalt_o2,block_size=(atrk_aggfac,),func=np.mean) ; obsalt_o2 = obsalt_o2 / norm_1d
                    time_o2 = block_reduce(time_o2,block_size=(atrk_aggfac,),func=np.mean) ; time_o2 = time_o2 / norm_1d
                    ac_alt_surf_o2 = block_reduce(ac_alt_surf_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_alt_surf_o2 = ac_alt_surf_o2 / norm_1d
                    ac_surf_alt_o2 = block_reduce(ac_surf_alt_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_surf_alt_o2 = ac_surf_alt_o2 / norm_1d
                    ac_lon_o2= block_reduce(ac_lon_o2,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lon_o2 = ac_lon_o2 / norm_1d
                    ac_lat_o2= block_reduce(ac_lat_o2,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lat_o2 = ac_lat_o2 / norm_1d
                    if(akaze_used==1):
                       ac_roll_o2    = block_reduce(ac_roll_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll_o2 = ac_roll_o2 / norm_1d 
                       ac_pitch_o2   = block_reduce(ac_pitch_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch_o2 = ac_pitch_o2 / norm_1d 
                       ac_heading_o2 = block_reduce(ac_heading_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading_o2 = ac_heading_o2 / norm_1d 
                       op_ac_lon_o2  = block_reduce(op_ac_lon_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lon_o2 = op_ac_lon_o2 / norm_1d
                       op_ac_lat_o2  = block_reduce(op_ac_lat_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lat_o2 = op_ac_lat_o2 / norm_1d
                       op_ac_alt_surf_o2  = block_reduce(op_ac_alt_surf_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_alt_surf_o2 = op_ac_alt_surf_o2/ norm_1d
                       op_ac_surf_alt_o2  = block_reduce(op_ac_surf_alt_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_surf_alt_o2 = op_ac_surf_alt_o2/ norm_1d
                       op_obsalt_o2  = block_reduce(op_obsalt_o2,block_size=(atrk_aggfac,),func=np.mean) ; op_obsalt_o2 = op_obsalt_o2 / norm_1d
                    elif(optimized_used==1):
                       ac_roll_o2    = block_reduce(ac_roll_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll_o2 = ac_roll_o2 / norm_1d 
                       ac_pitch_o2   = block_reduce(ac_pitch_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch_o2 = ac_pitch_o2 / norm_1d 
                       ac_heading_o2 = block_reduce(ac_heading_o2,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading_o2 = ac_heading_o2 / norm_1d 
                       av_ac_lon_o2 = block_reduce(av_ac_lon_o2,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lon_o2  =av_ac_lon_o2/ norm_1d
                       av_ac_lat_o2 = block_reduce(av_ac_lat_o2,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lat_o2  =av_ac_lat_o2/ norm_1d
                       av_ac_alt_surf_o2 = block_reduce(av_ac_alt_surf_o2,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_alt_surf_o2=av_ac_alt_surf_o2    / norm_1d
                       av_ac_surf_alt_o2 = block_reduce(av_ac_surf_alt_o2,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_surf_alt_o2=av_ac_surf_alt_o2    / norm_1d
                       av_obsalt_o2 = block_reduce(av_obsalt_o2,block_size=(atrk_aggfac,),func=np.mean) ;av_obsalt_o2 =av_obsalt_o2    / norm_1d
        
                    norm_2d = block_reduce(np.ones(lon_o2.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
                    lon_o2 = block_reduce(lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lon_o2 = lon_o2 / norm_2d
                    lat_o2 = block_reduce(lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lat_o2 = lat_o2 / norm_2d
                    saa_o2=block_reduce(saa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  saa_o2 = saa_o2 / norm_2d
                    sza_o2=block_reduce(sza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  sza_o2 = sza_o2 / norm_2d
                    vza_o2=block_reduce(vza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vza_o2 = vza_o2 / norm_2d
                    vaa_o2=block_reduce(vaa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vaa_o2 = vaa_o2 / norm_2d
                    aza_o2=block_reduce(aza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  aza_o2 = aza_o2 / norm_2d
                    surfalt_o2=block_reduce(surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  surfalt_o2 = surfalt_o2 / norm_2d
        
                    lon_o2 = lon_o2.transpose((1,0))
                    lat_o2 = lat_o2.transpose((1,0))
                    saa_o2=saa_o2.transpose((1,0))
                    sza_o2=sza_o2.transpose((1,0))
                    vza_o2=vza_o2.transpose((1,0))
                    vaa_o2=vaa_o2.transpose((1,0))
                    aza_o2=aza_o2.transpose((1,0))
                    surfalt_o2=surfalt_o2.transpose((1,0))
                    #ac_pos_o2=ac_pos_o2.transpose((1,0))
                    #ac_pix_bore_o2=ac_pix_bore_o2.transpose((2,1,0))
        
                    norm_3d = block_reduce(np.ones(ac_pix_bore_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    ac_pix_bore_o2=block_reduce(ac_pix_bore_o2,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ;  ac_pix_bore_o2 = ac_pix_bore_o2 / norm_3d
        
                    norm_2d = block_reduce(np.ones(ac_pos_o2.shape),(1,atrk_aggfac),func=np.mean)
                    ac_pos_o2=block_reduce(ac_pos_o2,block_size=(1,atrk_aggfac),func=np.mean) ;  ac_pos_o2 = ac_pos_o2 / norm_2d
        
                    norm_3d = block_reduce(np.ones(corner_lat_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                    corner_lat_o2 = block_reduce(corner_lat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lat_o2 = corner_lat_o2 / norm_3d
                    corner_lon_o2 = block_reduce(corner_lon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lon_o2 = corner_lon_o2 / norm_3d
        
                    if(akaze_used==1):
                        norm_2d = block_reduce(np.ones(akaze_msi_o2.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
                        akaze_msi_o2 = block_reduce(akaze_msi_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi_o2 = akaze_msi_o2 / norm_2d
                        akaze_Distance_Akaze_Reprojected_o2 = block_reduce(akaze_Distance_Akaze_Reprojected_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2 / norm_2d
                        av_lon_o2 = block_reduce(av_lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon_o2=av_lon_o2   / norm_2d
                        av_lat_o2 = block_reduce(av_lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat_o2=av_lat_o2   / norm_2d
                        av_surfalt_o2 = block_reduce(av_surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt_o2=av_surfalt_o2/norm_2d 
                        av_lon_o2 = av_lon_o2.transpose(1,0) 
                        av_lat_o2 = av_lat_o2.transpose(1,0) 
                        av_surfalt_o2 = av_surfalt_o2.transpose(1,0) 

                        op_lon_o2 = block_reduce(op_lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lon_o2=op_lon_o2  / norm_2d
                        op_lat_o2 = block_reduce(op_lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lat_o2=op_lat_o2  / norm_2d
                        op_sza_o2 = block_reduce(op_sza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_sza_o2=op_sza_o2  / norm_2d
                        op_saa_o2 = block_reduce(op_saa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_saa_o2=op_saa_o2  / norm_2d
                        op_aza_o2 = block_reduce(op_aza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_aza_o2=op_aza_o2  / norm_2d
                        op_vza_o2 = block_reduce(op_vza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vza_o2=op_vza_o2  / norm_2d
                        op_vaa_o2 = block_reduce(op_vaa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vaa_o2=op_vaa_o2  / norm_2d
                        op_surfalt_o2 = block_reduce(op_surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_surfalt_o2 =op_surfalt_o2 / norm_2d

                        op_lon_o2 = op_lon_o2.transpose(1,0) 
                        op_lat_o2 = op_lat_o2.transpose(1,0) 
                        op_sza_o2 = op_sza_o2.transpose(1,0) 
                        op_saa_o2 = op_saa_o2.transpose(1,0) 
                        op_aza_o2 = op_aza_o2.transpose(1,0) 
                        op_vza_o2 = op_vza_o2.transpose(1,0) 
                        op_vaa_o2 = op_vaa_o2.transpose(1,0) 
                        op_surfalt_o2 = op_surfalt_o2.transpose(1,0) 
                        akaze_msi_o2 = akaze_msi_o2.transpose(1,0)
                        akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2.transpose(1,0)



                        norm_3d = block_reduce(np.ones(av_clat_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)

                        av_clon_o2 = block_reduce(av_clon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon_o2 = av_clon_o2 / norm_3d 
                        av_clat_o2 = block_reduce(av_clat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat_o2 = av_clat_o2 / norm_3d 
                        op_clon_o2 = block_reduce(op_clon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clon_o2 = op_clon_o2 / norm_3d 
                        op_clat_o2 = block_reduce(op_clat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clat_o2 = op_clat_o2 / norm_3d 
                        op_clon_o2 = op_clon_o2.transpose((2,1,0))
                        op_clat_o2 = op_clat_o2.transpose((2,1,0))
                        av_clon_o2 = av_clon_o2.transpose((2,1,0))
                        av_clat_o2 = av_clat_o2.transpose((2,1,0))

                        norm_3d = block_reduce(np.ones(op_ac_pix_bore_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                        op_ac_pix_bore_o2 = block_reduce(op_ac_pix_bore_o2,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_ac_pix_bore_o2 = op_ac_pix_bore_o2/norm_3d 

                        norm_2d = block_reduce(np.ones(op_ac_pos_o2.shape),(1,atrk_aggfac),func=np.mean)
                        op_ac_pos_o2 = block_reduce(op_ac_pos_o2,block_size=(1,atrk_aggfac),func=np.mean) ;  op_ac_pos_o2 = op_ac_pos_o2 / norm_2d 

                        op_ac_pos_o2=op_ac_pos_o2.transpose((1,0))
                        op_ac_pix_bore_o2=op_ac_pix_bore_o2.transpose((2,1,0))
                    elif(optimized_used==1):
                        norm_2d = block_reduce(np.ones(akaze_msi_o2.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
                        akaze_lon_o2 = block_reduce(akaze_lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_lon_o2 = akaze_lon_o2 / norm_2d
                        akaze_lat_o2 = block_reduce(akaze_lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_lat_o2 = akaze_lat_o2 / norm_2d
                        akaze_msi_o2 = block_reduce(akaze_msi_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi_o2 = akaze_msi_o2 / norm_2d
                        akaze_surfalt_o2 = block_reduce(akaze_surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_surfalt_o2 = akaze_surfalt_o2 / norm_2d 
                        akaze_Distance_Akaze_Reprojected_o2 = block_reduce(akaze_Distance_Akaze_Reprojected_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2 / norm_2d

                        av_lon_o2 = block_reduce(av_lon_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon_o2=av_lon_o2   / norm_2d
                        av_lat_o2 = block_reduce(av_lat_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat_o2=av_lat_o2   / norm_2d
                        av_sza_o2 = block_reduce(av_sza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_sza_o2=av_sza_o2   / norm_2d
                        av_saa_o2 = block_reduce(av_saa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_saa_o2=av_saa_o2   / norm_2d
                        av_aza_o2 = block_reduce(av_aza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_aza_o2=av_aza_o2   / norm_2d
                        av_vza_o2 = block_reduce(av_vza_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vza_o2=av_vza_o2   / norm_2d
                        av_vaa_o2 = block_reduce(av_vaa_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vaa_o2=av_vaa_o2   / norm_2d
                        av_surfalt_o2 = block_reduce(av_surfalt_o2,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt_o2=av_surfalt_o2/norm_2d 
                        av_lon_o2 = av_lon_o2.transpose(1,0) 
                        av_lat_o2 = av_lat_o2.transpose(1,0) 
                        av_sza_o2 = av_sza_o2.transpose(1,0) 
                        av_saa_o2 = av_saa_o2.transpose(1,0) 
                        av_aza_o2 = av_aza_o2.transpose(1,0) 
                        av_vza_o2 = av_vza_o2.transpose(1,0) 
                        av_vaa_o2 = av_vaa_o2.transpose(1,0) 
                        av_surfalt_o2 = av_surfalt_o2.transpose(1,0) 


                        norm_3d = block_reduce(np.ones(akaze_clat_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                        akaze_clon_o2 = block_reduce(akaze_clon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clon_o2 = akaze_clon_o2 / norm_3d 
                        akaze_clat_o2 = block_reduce(akaze_clat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clat_o2 = akaze_clat_o2 / norm_3d 
                        av_clon_o2 = block_reduce(av_clon_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon_o2 = av_clon_o2 / norm_3d 
                        av_clat_o2 = block_reduce(av_clat_o2,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat_o2 = av_clat_o2 / norm_3d 
                        av_clon_o2 = av_clon_o2.transpose((2,1,0))
                        av_clat_o2 = av_clat_o2.transpose((2,1,0))
                        

                        akaze_lon_o2 = akaze_lon_o2.transpose(1,0)
                        akaze_lat_o2 = akaze_lat_o2.transpose(1,0)
                        akaze_msi_o2 = akaze_msi_o2.transpose(1,0)
                        akaze_surfalt_o2 = akaze_surfalt_o2.transpose(1,0)
                        akaze_Distance_Akaze_Reprojected_o2 = akaze_Distance_Akaze_Reprojected_o2.transpose(1,0)
                        akaze_clon_o2 = akaze_clon_o2.transpose((2,1,0))
                        akaze_clat_o2 = akaze_clat_o2.transpose((2,1,0))

                        norm_3d = block_reduce(np.ones(av_ac_pix_bore_o2.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
                        av_ac_pix_bore_o2 = block_reduce(av_ac_pix_bore_o2,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_ac_pix_bore_o2 = av_ac_pix_bore_o2/norm_3d 

                        norm_2d = block_reduce(np.ones(av_ac_pos_o2.shape),(1,atrk_aggfac),func=np.mean)
                        av_ac_pos_o2 = block_reduce(av_ac_pos_o2,block_size=(1,atrk_aggfac),func=np.mean) ;  av_ac_pos_o2 = av_ac_pos_o2 / norm_2d 

                        av_ac_pos_o2=av_ac_pos_o2.transpose((1,0))
                        av_ac_pix_bore_o2=av_ac_pix_bore_o2.transpose((2,1,0))
        
                    norm_3d = block_reduce(np.ones(rad_o2.shape),block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
                    valpix = np.zeros(rad_o2.shape)
                    idv = np.logical_and(np.isfinite(rad_o2),rad_o2>0.0)
                    valpix[idv] = 1.0
                    valpix_agg = block_reduce(valpix,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
                    valpix_agg = valpix_agg / norm_3d
        
                    rad_o2 = block_reduce(rad_o2,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_o2 = rad_o2 / norm_3d
                    rad_err_o2 = block_reduce(rad_err_o2,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_err_o2 = rad_err_o2 / norm_3d 
                    rad_err_o2 = rad_err_o2 / np.sqrt(xtrk_aggfac*atrk_aggfac)
                    rad_o2[valpix_agg <0.99999999999999999999] = np.nan
                    rad_err_o2[valpix_agg <0.99999999999999999999] = np.nan
        
                    wvl_o2 = block_reduce(wvl_o2,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; wvl_o2 = wvl_o2 / norm_3d
                    rad_flags_o2 = block_reduce(rad_flags_o2,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_flags_o2 = rad_flags_o2 / norm_3d
        
        
        
        
                    corner_lat_o2 = corner_lat_o2.transpose((2,1,0))
                    corner_lon_o2 = corner_lon_o2.transpose((2,1,0))
                    wvl_o2 = wvl_o2.transpose((2,1,0))
                    rad_o2 = rad_o2.transpose((2,1,0))
                    rad_err_o2 = rad_err_o2.transpose((2,1,0))
                    rad_flags_o2 = rad_flags_o2.transpose((2,1,0))
        
                    ac_pos_o2=ac_pos_o2.transpose((1,0))
                    ac_pix_bore_o2=ac_pix_bore_o2.transpose((2,1,0))
                    
                    l1 = pysplat.level1_AIR(l1_outfile,lon_o2,lat_o2,obsalt_o2,time_o2,ac_lon_o2,ac_lat_o2,ac_pos_o2,ac_surf_alt_o2,ac_alt_surf_o2,ac_pix_bore_o2,optbenchT=None,clon=corner_lon_o2,clat=corner_lat_o2,akaze_msi_image=akaze_msi_o2)
                    l1.set_2d_geofield('SurfaceAltitude', surfalt_o2)
                    l1.set_2d_geofield('SolarZenithAngle', sza_o2)
                    l1.set_2d_geofield('SolarAzimuthAngle', saa_o2)
                    l1.set_2d_geofield('ViewingZenithAngle', vza_o2)
                    l1.set_2d_geofield('ViewingAzimuthAngle', vaa_o2)
                    l1.set_2d_geofield('RelativeAzimuthAngle', aza_o2)
                    l1.add_radiance_band(wvl_o2,rad_o2,rad_err=rad_err_o2,rad_flag=rad_flags_o2)

                    if(optimized_used==1):

                        l1.set_supportfield('AkazeLongitude',akaze_lon_o2)
                        l1.set_supportfield('AkazeLatitude',akaze_lat_o2)
                        l1.set_supportfield('AkazeSurfaceAltitude',akaze_surfalt_o2)
                        l1.set_supportfield('AkazeCornerLatitude',akaze_clat_o2)
                        l1.set_supportfield('AkazeCornerLongitude',akaze_clon_o2)

                        l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt_o2)
                        l1.set_supportfield('AvionicsSolarZenithAngle',av_sza_o2)
                        l1.set_supportfield('AvionicsSolarAzimuthAngle',av_saa_o2)
                        l1.set_supportfield('AvionicsViewingZenithAngle',av_vza_o2)
                        l1.set_supportfield('AvionicsViewingAzimuthAngle',av_vaa_o2)
                        l1.set_supportfield('AvionicsRelativeAzimuthAngle',av_aza_o2)
                        l1.set_supportfield('AvionicsAircraftLongitude',av_ac_lon_o2)
                        l1.set_supportfield('AvionicsAircraftLatitude',av_ac_lat_o2)
                        l1.set_supportfield('AvionicsAircraftAltitudeAboveSurface',av_ac_alt_surf_o2)
                        l1.set_supportfield('AvionicsAircraftSurfaceAltitude',av_ac_surf_alt_o2)
                        l1.set_supportfield('AvionicsAircraftPixelBore',av_ac_pix_bore_o2)
                        l1.set_supportfield('AvionicsAircraftPos',av_ac_pos_o2)
                        l1.set_supportfield('AvionicsLongitude',av_lon_o2)
                        l1.set_supportfield('AvionicsLatitude',av_lat_o2)
                        l1.set_supportfield('AvionicsCornerLongitude',av_clon_o2)
                        l1.set_supportfield('AvionicsCornerLatitude',av_clat_o2)
                        l1.set_supportfield('AvionicsObservationAltitude',av_obsalt_o2)
                        l1.set_1d_flag(True,None,None)
                        l1.add_akaze(ac_roll_o2,ac_pitch_o2,ac_heading_o2,akaze_Distance_Akaze_Reprojected_o2,akaze_Optimization_Convergence_Fail_o2,akaze_Reprojection_Fit_Flag_o2)

                    elif(akaze_used==1):
                        l1.set_supportfield('OptimizedSolarZenithAngle', op_sza_o2)
                        l1.set_supportfield('OptimizedSolarAzimuthAngle', op_saa_o2)
                        l1.set_supportfield('OptimizedViewingZenithAngle', op_vza_o2)
                        l1.set_supportfield('OptimizedViewingAzimuthAngle', op_vaa_o2)
                        l1.set_supportfield('OptimizedRelativeAzimuthAngle', op_aza_o2)
                        l1.set_supportfield('OptimizedSurfaceAltitude',op_surfalt_o2)
                        l1.set_supportfield('OptimizedAircraftLongitude',op_ac_lon_o2)
                        l1.set_supportfield('OptimizedAircraftLatitude',op_ac_lat_o2)
                        l1.set_supportfield('OptimizedAircraftAltitudeAboveSurface',op_ac_alt_surf_o2)
                        l1.set_supportfield('OptimizedAircraftSurfaceAltitude',op_ac_surf_alt_o2)
                        l1.set_supportfield('OptimizedAircraftPixelBore',op_ac_pix_bore_o2)
                        l1.set_supportfield('OptimizedAircraftPos',op_ac_pos_o2)
                        l1.set_supportfield('OptimizedLongitude',op_lon_o2)
                        l1.set_supportfield('OptimizedLatitude',op_lat_o2)
                        l1.set_supportfield('OptimizedCornerLongitude',op_clon_o2)
                        l1.set_supportfield('OptimizedCornerLatitude',op_clat_o2)
                        l1.set_supportfield('OptimizedObservationAltitude',op_obsalt_o2)

                        l1.set_supportfield('AvionicsLongitude',av_lon_o2)
                        l1.set_supportfield('AvionicsLatitude',av_lat_o2)
                        l1.set_supportfield('AvionicsCornerLongitude',av_clon_o2)
                        l1.set_supportfield('AvionicsCornerLatitude',av_clat_o2)
                        l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt_o2)

                        l1.set_1d_flag(None,True,True)
                        l1.add_akaze(ac_roll_o2,ac_pitch_o2,ac_heading_o2,akaze_Distance_Akaze_Reprojected_o2,akaze_Optimization_Convergence_Fail_o2,akaze_Reprojection_Fit_Flag_o2)
                    else:
                        l1.set_1d_flag(None,None,True)


                    l1.close()
                    ncfile = nc4.Dataset(l1_outfile,'a',format="NETCDF4")
                    ncgroup = ncfile.createGroup('AlignmentMetaData') 
                    data = ncgroup.createVariable('NativeXtrackShift',np.int16,('o'))
                    data[:] = total_shift_in_pixels
                    ncfile.close()
                      
        
                    lockname=logfile_o2+'.lock'
                    with FileLock(lockname):
                        f = open(logfile_o2,'a+') 
                        f.write(str(l1_outfile)+' '+str(priority)+'\n' )
                        f.close()
                  
                if(fileo2 is not None):
                    fig, ax = plt.subplots(2,2)
                    plt.rcParams.update({'font.size': 8})
        
                    ax[0,0].imshow(datao2.T, interpolation='none', origin='lower', cmap='Greys_r', aspect='auto')
                    ax[0,1].imshow(datach4.T, interpolation='none', origin='lower', cmap='Greys_r', aspect='auto')
                    
                    ax[0,0].set_ylabel('Xtrack Index')
                    
                    ax[0,0].set_title('Old O2 Radiance')
                    ax[0,1].set_title('Old CH4 Radiance')
        
                    ax[1,1].imshow((datao2+ch4_new).T,interpolation='none',origin='lower',  cmap='Greys_r',aspect='auto')
                    ax[1,1].set_title('(New CH4 + O2) Radiance')
                    ax[1,1].set_xlabel('Along Track Index')
                    ax[1,0].imshow(ch4_new.T, interpolation='none',origin='lower',  cmap='Greys_r', aspect='auto')
                    ax[1,0].set_xlabel('Along Track Index')
                    ax[1,0].set_ylabel('Xtrack Index')
                    ax[1,0].set_title('New CH4 Radiance')
                    
        
                    fig.subplots_adjust(hspace=0.4)
        
                    filename = os.path.join('Final_'+self.inputCH4Times+'.png')
                    plt.savefig(filename,dpi=1000)
                    plt.close()
                else:
                    fig, ax = plt.subplots(2)
                    plt.rcParams.update({'font.size': 8})
        
                    ax[0].set_title('Old CH4 Radiance')
                    ax[0].set_ylabel('Xtrack Index')
                    ax[0].imshow(datach4.T, interpolation='none', origin='lower', cmap='Greys_r', aspect='auto')
                    
                    ax[1].imshow(ch4_new.T, interpolation='none',origin='lower',  cmap='Greys_r', aspect='auto')
                    ax[1].set_xlabel('Along Track Index')
                    ax[1].set_ylabel('Xtrack Index')
                    ax[1].set_title('New CH4 Radiance')
                    
        
                    fig.subplots_adjust(hspace=0.4)
        
                    filename = os.path.join('Final_'+self.inputCH4Times+'.png')
                    plt.savefig(filename,dpi=1000)
                    plt.close()
                print('Time = ',(time.time() - self.t0)) 

    def assign_data_optimized(self,rad,raderr,wvl,radflag,lon,lat,clon,clat,vza,sza,vaa,saa,aza,obsalt,surfalt,time,ac_lon,ac_lat,ac_alt_surf,ac_surf_alt,ac_pix_bore,ac_pos,akz_lon,akz_lat,akz_msi,akaze_clon,akaze_clat,akaze_surfalt,akaze_Distance_Akaze_Reprojected,akaze_Optimization_Convergence_Fail,akaze_Reprojection_Fit_Flag,av_lon,av_lat,av_clon,av_clat,av_sza,av_saa,av_aza,av_vza,av_vaa,av_surfalt,av_ac_lon,av_ac_lat,av_ac_alt_surf,av_ac_surf_alt,av_ac_pix_bore,av_ac_pos,av_obsalt,akaze_used,optimized_used,avionics_used,ac_roll,ac_pitch,ac_heading,a1,a2,b1,b2,c1,c2,d1,d2):

        wvl[:,a1:a2,b1:b2] = self.x1['Wavelength'][:,c1:c2,d1:d2]
        raderr[:,a1:a2,b1:b2] = self.x1['RadianceUncertainty'][:,c1:c2,d1:d2]
        radflag[:,a1:a2,b1:b2] = self.x1['RadianceFlag'][:,c1:c2,d1:d2]
        radflag[:,a1:a2,b1:b2] = self.x1['RadianceFlag'][:,c1:c2,d1:d2]
        rad[:,a1:a2,b1:b2] = self.x1['Radiance'][:,c1:c2,d1:d2]
    
        
        clon[:,a1:a2,b1:b2] = self.x2['CornerLongitude'][:,c1:c2,d1:d2]
        clat[:,a1:a2,b1:b2] = self.x2['CornerLatitude'][:,c1:c2,d1:d2]
        lon[a1:a2,b1:b2] = self.x2['Longitude'][c1:c2,d1:d2]
        lat[a1:a2,b1:b2] = self.x2['Latitude'][c1:c2,d1:d2]
        sza[a1:a2,b1:b2] = self.x2['SolarZenithAngle'][c1:c2,d1:d2]
        vza[a1:a2,b1:b2] = self.x2['ViewingZenithAngle'][c1:c2,d1:d2]
        vaa[a1:a2,b1:b2] = self.x2['ViewingAzimuthAngle'][c1:c2,d1:d2]
        saa[a1:a2,b1:b2] = self.x2['SolarAzimuthAngle'][c1:c2,d1:d2]
        aza[a1:a2,b1:b2] = self.x2['RelativeAzimuthAngle'][c1:c2,d1:d2]
        surfalt[a1:a2] = self.x2['SurfaceAltitude'][c1:c2]
        obsalt[a1:a2] = self.x2['ObservationAltitude'][c1:c2]
        time[a1:a2] = self.x2['Time'][c1:c2]
        ac_lon[a1:a2] = self.x2['AircraftLongitude'][c1:c2]
        ac_lat[a1:a2] = self.x2['AircraftLatitude'][c1:c2]
        ac_alt_surf[a1:a2] = self.x2['AircraftAltitudeAboveSurface'][c1:c2]
        ac_surf_alt[a1:a2] = self.x2['AircraftSurfaceAltitude'][c1:c2]
        ac_pix_bore[:,a1:a2,b1:b2] = self.x2['AircraftPixelBore'][:,c1:c2,d1:d2]
        ac_pos[:,a1:a2] = self.x2['AircraftPos'][:,c1:c2]

        akz_lon[a1:a2,b1:b2] = self.x3['AkazeLongitude'][c1:c2,d1:d2]
        akz_lat[a1:a2,b1:b2] = self.x3['AkazeLatitude'][c1:c2,d1:d2]
        akz_msi[a1:a2,b1:b2] = self.x3['AkazeMSIImage'][c1:c2,d1:d2]
        akaze_clon[:,a1:a2,b1:b2] = self.x3['AkazeCornerLongitude'][:,c1:c2,d1:d2]
        akaze_clat[:,a1:a2,b1:b2] = self.x3['AkazeCornerLatitude'][:,c1:c2,d1:d2]
        akaze_surfalt[a1:a2,b1:b2] = self.x3['AkazeSurfaceAltitude'][c1:c2,d1:d2]

        akaze_Distance_Akaze_Reprojected[a1:a2,b1:b2] = self.x3['DistanceAkazeReprojected'][c1:c2,d1:d2]
        akaze_Optimization_Convergence_Fail[:] = self.x3['OptimizationConvergenceFail'][:]
        akaze_Reprojection_Fit_Flag[:] = self.x3['ReprojectionFitFlag'][:]
        # Add avionics variables
        av_lon[a1:a2,b1:b2] = self.x3['AvionicsLongitude'][c1:c2,d1:d2]
        av_lat[a1:a2,b1:b2] = self.x3['AvionicsLatitude'][c1:c2,d1:d2]
        av_clon[:,a1:a2,b1:b2] = self.x3['AvionicsCornerLongitude'][:,c1:c2,d1:d2]
        av_clat[:,a1:a2,b1:b2] = self.x3['AvionicsCornerLatitude'][:,c1:c2,d1:d2]
        av_sza[a1:a2,b1:b2] = self.x3['AvionicsSolarZenithAngle'][c1:c2,d1:d2]
        av_saa[a1:a2,b1:b2] = self.x3['AvionicsSolarAzimuthAngle'][c1:c2,d1:d2] 
        av_aza[a1:a2,b1:b2] = self.x3['AvionicsRelativeAzimuthAngle'][c1:c2,d1:d2]  
        av_vza[a1:a2,b1:b2] = self.x3['AvionicsViewingZenithAngle'][c1:c2,d1:d2] 
        av_vaa[a1:a2,b1:b2] = self.x3['AvionicsViewingAzimuthAngle'][c1:c2,d1:d2]  
        av_surfalt[a1:a2,b1:b2] = self.x3['AvionicsSurfaceAltitude'][c1:c2,d1:d2]  
        av_ac_lon[a1:a2] = self.x3['AvionicsAircraftLongitude'][c1:c2] 
        av_ac_lat[a1:a2] = self.x3['AvionicsAircraftLatitude'][c1:c2] 
        av_ac_alt_surf[a1:a2] = self.x3['AvionicsAircraftAltitudeAboveSurface'][c1:c2] 
        av_ac_surf_alt[a1:a2] = self.x3['AvionicsAircraftSurfaceAltitude'][c1:c2]  
        av_ac_pix_bore[:,a1:a2,b1:b2] = self.x3['AvionicsAircraftPixelBore'][:,c1:c2,d1:d2]  
        av_ac_pos[:,a1:a2] = self.x3['AvionicsAircraftPos'][:,c1:c2]
        av_obsalt[a1:a2] = self.x3['AvionicsObservationAltitude'][c1:c2]  
        akaze_used = self.x3['AkazeUsed'][0]
        optimized_used = self.x3['OptimizedUsed'][0]
        avionics_used = self.x3['AvionicsUsed'][0]
        ac_roll[a1:a2] = self.x3['AircraftRoll'][c1:c2]
        ac_pitch[a1:a2] = self.x3['AircraftPitch'][c1:c2]
        ac_heading[a1:a2] = self.x3['AircraftHeading'][c1:c2]
        #for i in range(a1,a2):
        #    for j in range(b1,b2):
        #        aza[i,j] = vaa[i,j] - (180.0 - saa[i,j])
        #        if( aza[i,j] < 0.0 ):
        #            aza[i,j] = aza[i,j] + 360.0
        #        if( aza[i,j] > 360.0 ):
        #            aza[i,j] = aza[i,j] - 360.0
    
        return (rad,raderr,wvl,radflag,lon,lat,clon,clat,vza,sza,vaa,saa,aza,obsalt,surfalt,time,ac_lon,ac_lat,ac_alt_surf,ac_surf_alt,ac_pix_bore,ac_pos,akz_lon,akz_lat,akz_msi,akaze_clon,akaze_clat,akaze_surfalt,akaze_Distance_Akaze_Reprojected,akaze_Optimization_Convergence_Fail,akaze_Reprojection_Fit_Flag,av_lon,av_lat,av_clon,av_clat,av_sza,av_saa,av_aza,av_vza,av_vaa,av_surfalt,av_ac_lon,av_ac_lat,av_ac_alt_surf,av_ac_surf_alt,av_ac_pix_bore,av_ac_pos,av_obsalt,akaze_used,optimized_used,avionics_used,ac_roll,ac_pitch,ac_heading)






    def assign_data_akaze(self,rad,raderr,wvl,radflag,lon,lat,clon,clat,vza,sza,vaa,saa,aza,obsalt,surfalt,time,ac_lon,ac_lat,ac_alt_surf,ac_surf_alt,ac_pix_bore,ac_pos,akz_msi,av_lon,av_lat,av_clon,av_clat,av_surfalt,op_lon,op_lat,op_clon,op_clat,op_sza,op_saa,op_aza,op_vza,op_vaa,op_surfalt,op_ac_lon,op_ac_lat,op_ac_alt_surf,op_ac_surf_alt,op_ac_pix_bore,op_ac_pos,op_obsalt,akaze_Distance_Akaze_Reprojected,akaze_Optimization_Convergence_Fail,akaze_Reprojection_Fit_Flag,akaze_used,optimized_used,avionics_used,ac_roll,ac_pitch,ac_heading,a1,a2,b1,b2,c1,c2,d1,d2):


        wvl[:,a1:a2,b1:b2] = self.x1['Wavelength'][:,c1:c2,d1:d2]
        raderr[:,a1:a2,b1:b2] = self.x1['RadianceUncertainty'][:,c1:c2,d1:d2]
        radflag[:,a1:a2,b1:b2] = self.x1['RadianceFlag'][:,c1:c2,d1:d2]
        radflag[:,a1:a2,b1:b2] = self.x1['RadianceFlag'][:,c1:c2,d1:d2]
        rad[:,a1:a2,b1:b2] = self.x1['Radiance'][:,c1:c2,d1:d2]
    
        
        clon[:,a1:a2,b1:b2] = self.x2['CornerLongitude'][:,c1:c2,d1:d2]
        clat[:,a1:a2,b1:b2] = self.x2['CornerLatitude'][:,c1:c2,d1:d2]
        lon[a1:a2,b1:b2] = self.x2['Longitude'][c1:c2,d1:d2]
        lat[a1:a2,b1:b2] = self.x2['Latitude'][c1:c2,d1:d2]
        sza[a1:a2,b1:b2] = self.x2['SolarZenithAngle'][c1:c2,d1:d2]
        vza[a1:a2,b1:b2] = self.x2['ViewingZenithAngle'][c1:c2,d1:d2]
        vaa[a1:a2,b1:b2] = self.x2['ViewingAzimuthAngle'][c1:c2,d1:d2]
        saa[a1:a2,b1:b2] = self.x2['SolarAzimuthAngle'][c1:c2,d1:d2]
        aza[a1:a2,b1:b2] = self.x2['RelativeAzimuthAngle'][c1:c2,d1:d2]
        surfalt[a1:a2] = self.x2['SurfaceAltitude'][c1:c2]
        obsalt[a1:a2] = self.x2['ObservationAltitude'][c1:c2]
        time[a1:a2] = self.x2['Time'][c1:c2]
        ac_lon[a1:a2] = self.x2['AircraftLongitude'][c1:c2]
        ac_lat[a1:a2] = self.x2['AircraftLatitude'][c1:c2]
        ac_alt_surf[a1:a2] = self.x2['AircraftAltitudeAboveSurface'][c1:c2]
        ac_surf_alt[a1:a2] = self.x2['AircraftSurfaceAltitude'][c1:c2]
        ac_pix_bore[:,a1:a2,b1:b2] = self.x2['AircraftPixelBore'][:,c1:c2,d1:d2]
        ac_pos[:,a1:a2] = self.x2['AircraftPos'][:,c1:c2]

        akz_msi[a1:a2,b1:b2] = self.x3['AkazeMSIImage'][c1:c2,d1:d2]
        av_lon[a1:a2,b1:b2] = self.x3['AvionicsLongitude'][c1:c2,d1:d2] 
        av_lat[a1:a2,b1:b2] = self.x3['AvionicsLatitude'][c1:c2,d1:d2]
        av_clon[:,a1:a2,b1:b2] = self.x3['AvionicsCornerLongitude'][:,c1:c2,d1:d2]
        av_clat[:,a1:a2,b1:b2] = self.x3['AvionicsCornerLatitude'][:,c1:c2,d1:d2]
        av_surfalt[a1:a2,b1:b2] = self.x3['AvionicsSurfaceAltitude'][c1:c2,d1:d2] 
        # Add all optimized variables
        op_lon[a1:a2,b1:b2] = self.x3['OptimizedLongitude'][c1:c2,d1:d2] 
        op_lat[a1:a2,b1:b2] = self.x3['OptimizedLatitude'][c1:c2,d1:d2]
        op_clon[:,a1:a2,b1:b2] = self.x3['OptimizedCornerLongitude'][:,c1:c2,d1:d2]
        op_clat[:,a1:a2,b1:b2] = self.x3['OptimizedCornerLatitude'][:,c1:c2,d1:d2]
        op_sza[a1:a2,b1:b2] = self.x3['OptimizedSolarZenithAngle'][c1:c2,d1:d2]
        op_saa[a1:a2,b1:b2] = self.x3['OptimizedSolarAzimuthAngle'][c1:c2,d1:d2]
        op_aza[a1:a2,b1:b2] = self.x3['OptimizedRelativeAzimuthAngle'][c1:c2,d1:d2] 
        op_vza[a1:a2,b1:b2] = self.x3['OptimizedViewingZenithAngle'][c1:c2,d1:d2]
        op_vaa[a1:a2,b1:b2] = self.x3['OptimizedViewingAzimuthAngle'][c1:c2,d1:d2] 
        op_surfalt[a1:a2,b1:b2] = self.x3['OptimizedSurfaceAltitude'][c1:c2,d1:d2] 
        op_ac_lon[a1:a2] = self.x3['OptimizedAircraftLongitude'][c1:c2]
        op_ac_lat[a1:a2] = self.x3['OptimizedAircraftLatitude'][c1:c2]
        op_ac_alt_surf[a1:a2] = self.x3['OptimizedAircraftAltitudeAboveSurface'][c1:c2]
        op_ac_surf_alt[a1:a2] = self.x3['OptimizedAircraftSurfaceAltitude'][c1:c2]
        op_ac_pix_bore[:,a1:a2,b1:b2] = self.x3['OptimizedAircraftPixelBore'][:,c1:c2,d1:d2]
        op_ac_pos[:,a1:a2] = self.x3['OptimizedAircraftPos'][:,c1:c2]
        op_obsalt[a1:a2] = self.x3['OptimizedObservationAltitude'][c1:c2] 
        akaze_Distance_Akaze_Reprojected[a1:a2,b1:b2] = self.x3['DistanceAkazeReprojected'][c1:c2,d1:d2]
        akaze_Optimization_Convergence_Fail[:] = self.x3['OptimizationConvergenceFail'][:] 
        akaze_Reprojection_Fit_Flag[:] = self.x3['ReprojectionFitFlag'][:]
        akaze_used = self.x3['AkazeUsed'][0]
        optimized_used = self.x3['OptimizedUsed'][0]
        avionics_used = self.x3['AvionicsUsed'][0]
        ac_roll[a1:a2] = self.x3['AircraftRoll'][c1:c2]
        ac_pitch[a1:a2] = self.x3['AircraftPitch'][c1:c2]
        ac_heading[a1:a2] = self.x3['AircraftHeading'][c1:c2]
    
        #for i in range(a1,a2):
        #    for j in range(b1,b2):
        #        aza[i,j] = vaa[i,j] - (180.0 - saa[i,j])
        #        if( aza[i,j] < 0.0 ):
        #            aza[i,j] = aza[i,j] + 360.0
        #        if( aza[i,j] > 360.0 ):
        #            aza[i,j] = aza[i,j] - 360.0
    
        return (rad,raderr,wvl,radflag,lon,lat,clon,clat,vza,sza,vaa,saa,aza,obsalt,surfalt,time,ac_lon,ac_lat,ac_alt_surf,ac_surf_alt,ac_pix_bore,ac_pos,akz_msi,av_lon,av_lat,av_clon,av_clat,av_surfalt,op_lon,op_lat,op_clon,op_clat,op_sza,op_saa,op_aza,op_vza,op_vaa,op_surfalt,op_ac_lon,op_ac_lat,op_ac_alt_surf,op_ac_surf_alt,op_ac_pix_bore,op_ac_pos,op_obsalt,akaze_Distance_Akaze_Reprojected,akaze_Optimization_Convergence_Fail,akaze_Reprojection_Fit_Flag,akaze_used,optimized_used,avionics_used,ac_roll,ac_pitch,ac_heading)


    def assign_data(self,rad,raderr,wvl,radflag,lon,lat,clon,clat,vza,sza,vaa,saa,aza,obsalt,surfalt,time,ac_lon,ac_lat,ac_alt_surf,ac_surf_alt,ac_pix_bore,ac_pos,a1,a2,b1,b2,c1,c2,d1,d2):
        wvl[:,a1:a2,b1:b2] = self.x1['Wavelength'][:,c1:c2,d1:d2]
        raderr[:,a1:a2,b1:b2] = self.x1['RadianceUncertainty'][:,c1:c2,d1:d2]
        radflag[:,a1:a2,b1:b2] = self.x1['RadianceFlag'][:,c1:c2,d1:d2]
        radflag[:,a1:a2,b1:b2] = self.x1['RadianceFlag'][:,c1:c2,d1:d2]
        rad[:,a1:a2,b1:b2] = self.x1['Radiance'][:,c1:c2,d1:d2]
    
        
        clon[:,a1:a2,b1:b2] = self.x2['CornerLongitude'][:,c1:c2,d1:d2]
        clat[:,a1:a2,b1:b2] = self.x2['CornerLatitude'][:,c1:c2,d1:d2]
        lon[a1:a2,b1:b2] = self.x2['Longitude'][c1:c2,d1:d2]
        lat[a1:a2,b1:b2] = self.x2['Latitude'][c1:c2,d1:d2]
        sza[a1:a2,b1:b2] = self.x2['SolarZenithAngle'][c1:c2,d1:d2]
        vza[a1:a2,b1:b2] = self.x2['ViewingZenithAngle'][c1:c2,d1:d2]
        vaa[a1:a2,b1:b2] = self.x2['ViewingAzimuthAngle'][c1:c2,d1:d2]
        saa[a1:a2,b1:b2] = self.x2['SolarAzimuthAngle'][c1:c2,d1:d2]
        aza[a1:a2,b1:b2] = self.x2['RelativeAzimuthAngle'][c1:c2,d1:d2]
        surfalt[a1:a2] = self.x2['SurfaceAltitude'][c1:c2]
        obsalt[a1:a2] = self.x2['ObservationAltitude'][c1:c2]
        time[a1:a2] = self.x2['Time'][c1:c2]
        ac_lon[a1:a2] = self.x2['AircraftLongitude'][c1:c2]
        ac_lat[a1:a2] = self.x2['AircraftLatitude'][c1:c2]
        ac_alt_surf[a1:a2] = self.x2['AircraftAltitudeAboveSurface'][c1:c2]
        ac_surf_alt[a1:a2] = self.x2['AircraftSurfaceAltitude'][c1:c2]
        ac_pix_bore[:,a1:a2,b1:b2] = self.x2['AircraftPixelBore'][:,c1:c2,d1:d2]
        ac_pos[:,a1:a2] = self.x2['AircraftPos'][:,c1:c2]
    
        #for i in range(a1,a2):
        #    for j in range(b1,b2):
        #        aza[i,j] = vaa[i,j] - (180.0 - saa[i,j])
        #        if( aza[i,j] < 0.0 ):
        #            aza[i,j] = aza[i,j] + 360.0
        #        if( aza[i,j] > 360.0 ):
        #            aza[i,j] = aza[i,j] - 360.0
    
        return (rad,raderr,wvl,radflag,lon,lat,clon,clat,vza,sza,vaa,saa,aza,obsalt,surfalt,time,ac_lon,ac_lat,ac_alt_surf,ac_surf_alt,ac_pix_bore,ac_pos)
