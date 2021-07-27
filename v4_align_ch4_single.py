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
import os
import pysplat
from skimage.measure import block_reduce




class Alignment(object):

    def __init__(self,inputfile):
        logfile_ch4_native = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02/EKC_V4_with_Stray/CH4_NATIVE/log_file.txt' 
        g = open(logfile_ch4_native,'r')
        native = g.readlines()  
        g.close()
        nfiles = len(native)
         
        native_ch4_files = []
        native_ch4_priority = []
        
        for i in range(nfiles):
            native_ch4_files.append(os.path.basename(native[i].split(' ')[0]))
            native_ch4_priority.append((native[i].split(' ')[1]).split('\n')[0])

        #logfile_o2_native = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02/EKC_V4_with_Stray/O2_NATIVE/log_file.txt' 
        #g = open(logfile_o2_native,'r')
        #native = g.readlines()  
        #g.close()
        #nfiles = len(native)
        # 
        #native_o2_files = []
        #native_o2_priority = []
        #
        #for i in range(nfiles):
        #    native_o2_files.append(os.path.basename(native[i].split(' ')[0]))
        #    native_o2_priority.append((native[i].split(' ')[1]).split('\n')[0])
       

        method = cv2.TM_SQDIFF_NORMED
        ### GET THE LIST OF CH4 FILES FIRST
        ch4dir = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02/EKC_V4_with_Stray/CH4_NATIVE/'
        CH4Files = []
        CH4Times = []
        CH4StartTime = []
        CH4EndTime = []
        count=0
        for file in os.listdir(ch4dir):
            if file.endswith(".nc"):
                count=count+1
                base=os.path.basename(file)
                name=(base.split(".nc"))[0]
                start = name[28:34]
                end = name[44:50]
                name = name[19:66]
                CH4Files.append(os.path.join(ch4dir, file))
                CH4Times.append(name)
                CH4EndTime.append(end)
                CH4StartTime.append(start)
    
    
        ### GET THE LIST OF O2 FILES NEXT
        o2dir = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02/EKC_V4_with_Stray/O2_NATIVE/'
        O2Files = []
        O2Times = []
        O2StartTime = []
        O2EndTime = []
        for file in os.listdir(o2dir):
            count=count+1
            if file.endswith(".nc"):
                base=os.path.basename(file)
                name=(base.split(".nc"))[0]
                start = name[27:33]
                end = name[43:49]
                name = name[18:65]
                O2Files.append(os.path.join(o2dir, file))
                O2Times.append(name)
                O2EndTime.append(end)
                O2StartTime.append(start)

        inputCH4Times = []
        inputCH4StartTime = []
        inputCH4EndTime = []
        base=os.path.basename(inputfile)
        name=(base.split(".nc"))[0]
        start = name[28:34]
        end = name[44:50]
        name = name[19:66]
        inputCH4Times=name
        inputCH4EndTime=end
        inputCH4StartTime=start
        
        hseconds = abs(np.int(inputCH4EndTime)) % 100
        hminutes = ((abs(np.int(inputCH4EndTime)) % 10000) - hseconds)/100
        hhour = (np.int(inputCH4EndTime) - hseconds - hminutes*100 ) / 10000 
        delta = np.int(inputCH4EndTime) - np.int(inputCH4StartTime)

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
            for j in range(len(O2Times)):
                starto2seconds =  datetime.datetime.strptime(O2StartTime[j],"%H%M%S").second
                starto2minutes = datetime.datetime.strptime(O2StartTime[j],"%H%M%S").minute
                starto2hour = datetime.datetime.strptime(O2StartTime[j],"%H%M%S").hour

                endo2seconds =  datetime.datetime.strptime(O2EndTime[j],"%H%M%S").second
                endo2minutes = datetime.datetime.strptime(O2EndTime[j],"%H%M%S").minute
                endo2hour = datetime.datetime.strptime(O2EndTime[j],"%H%M%S").hour

                #if((inputCH4StartTime == O2StartTime[j])   and (  inputCH4EndTime == O2EndTime[j]  )   ):
                if((math.isclose(startch4seconds,starto2seconds,abs_tol=1.1)) and (math.isclose(endch4seconds,endo2seconds,abs_tol=1.1)) and 
                     (startch4minutes == starto2minutes) and (startch4hour == starto2hour) 
                    and (endch4minutes == endo2minutes) and (endch4hour == endo2hour) ):
        
                    #CH4
                    filech4 = inputfile 
                    datach4 = Dataset(filech4)
                    self.x1 = datach4.groups['Band1']
                    self.x2 = datach4.groups['Geolocation']
                    radch4 = self.x1['Radiance'][:,:,:].data
        
                    #O2
                    fileo2 = O2Files[j] 
                    datao2 = Dataset(fileo2)
                    y1 = datao2.groups['Band1']
                    y2 = datao2.groups['Geolocation']
                    rado2 = y1['Radiance'][:,:,:].data
        
        
        
                    wvl_new = np.zeros(y1['Wavelength'].shape )
                    rad_err_new = np.zeros(y1['RadianceUncertainty'].shape)
                    rad_flags_new = np.zeros(y1['RadianceFlag'].shape)
                    rad_new = np.zeros(y1['Radiance'].shape   )
                    
                    corner_lon_new = np.zeros(y2['CornerLongitude'].shape)
                    corner_lat_new = np.zeros(y2['CornerLatitude'].shape)
                    lon_new = np.zeros(y2['Longitude'].shape                            )
                    lat_new = np.zeros(y2['Latitude'].shape                             )
                    sza_new = np.zeros(y2['SolarZenithAngle'].shape        )
                    SolarAzimuthAngle_new = np.zeros(y2['SolarAzimuthAngle'].shape        )
                    vza_new = np.zeros(y2['ViewingZenithAngle'].shape    )
                    vaa_new = np.zeros(y2['ViewingAzimuthAngle'].shape  )
                    saa_new = np.zeros(y2['SolarAzimuthAngle'].shape  )
                    aza_new = np.zeros(y2['RelativeAzimuthAngle'].shape)
                    surfalt_new = np.zeros(y2['SurfaceAltitude'].shape)
                    obsalt_new = np.zeros(y2['ObservationAltitude'].shape)
                    time_new = np.zeros(y2['Time'].shape)
                    ac_lon_new = np.zeros(y2['Aircraft_Longitude'].shape)
                    ac_lat_new = np.zeros(y2['Aircraft_Latitude'].shape)
                    ac_alt_surf_new = np.zeros(y2['Aircraft_AltitudeAboveSurface'].shape)
                    ac_surf_alt_new = np.zeros(y2['Aircraft_SurfaceAltitude'].shape)
                    ac_pix_bore_new = np.zeros(y2['Aircraft_PixelBore'].shape)
                    ac_pos_new = np.zeros(y2['Aircraft_Pos'].shape)
        
        
                    wvl_new.fill(np.nan)
                    rad_err_new.fill(np.nan)
                    rad_flags_new.fill(np.nan)
                    rad_new.fill(np.nan) 
                    
                    corner_lon_new.fill(np.nan) 
                    corner_lat_new.fill(np.nan)
                    lon_new.fill(np.nan) 
                    lat_new.fill(np.nan) 
                    sza_new.fill(np.nan) 
                    SolarAzimuthAngle_new.fill(np.nan) 
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
                            if((abs(xshift) > 315) or (abs(xshift) < 295)):
                                print('Xtrack Shift fails')
                                print('Assigning Xtrack shift = 304 : the default')
                                xshift = 304 

                        elif(x == 1):  
                            print('Very few frames:')
                            print('Assigning Atrack shift = 0 : the default')
                            print('Assigning Xtrack shift = 304 : the default')
                            ashift = 0 
                            xshift = 304 

                        if( (xshift >= 295) and (xshift <= 315) ): 
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
                
                                print('File ',k,' of ',len(CH4Times) )
                                print('FileTime = ',inputfile )
                                print('ch4 shape = ',datach4.shape)
                                print('o2 shape = ',datao2.shape)
                                print('xshift = ',xshift)
                                print('ashift = ',ashift)
                
                
                
                                ch4_new = np.zeros(( (datao2.shape[0]  ) ,  (datach4.shape[1]  )   ))
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
                                      
                                   
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    
                                    ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
        
                                    for m in range(len(CH4Times)):
        
                                        mseconds = abs(np.int(CH4StartTime[m])) % 100
                                        mminutes = ((abs(np.int(CH4StartTime[m])) % 10000) - mseconds)/100
                                        mhour = (np.int(CH4StartTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                                        if(delta != np.int(9)):
                                            if(np.int(inputCH4EndTime)%10 != 9): 
                                                timediff = 0 
                                            else: 
                                                timediff = 1 
                                        else:
                                            timediff = 1
                                        if( (np.int(hseconds) == 59   ) and (np.int(hminutes) == 59   )  ): # change hour and minute and second of m file 
                                            if( ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(hhour + 1)   )) or  ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(hhour )   ))    ):  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
        
                                        elif( (np.int(hseconds) == 59   ) and (np.int(hminutes) != 59   )  ): #  minute and second of m file V
                                            if( ((np.int(mseconds) == 00   ) and (np.int(mminutes) == (np.int(hminutes) + 1)   ) and (np.int(mhour) == np.int(hhour )   ) ) or  ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(hminutes) )   ) and (np.int(mhour) == np.int(hhour )   ) )   ):  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                    
                                        elif( (np.int(hseconds) != 59   ) and (np.int(hminutes) == 59   )  ): #  minute and second of m file V
                                            if( math.isclose(np.int(mseconds) ,np.int(hseconds),abs_tol=1 )    and (np.int(mminutes) == (np.int(hminutes) )   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                                ch4_new[np.int(a1):np.int(a2),np.int(b1):np.int(b2)] = datach4[np.int(c1):np.int(c2),np.int(d1):np.int(d2)]
                                        elif( (np.int(hseconds) != 59   ) and (np.int(hminutes) != 59   )  ): #  minute and second of m file V
                                            if( math.isclose(np.int(mseconds) , np.int(hseconds),abs_tol=1.1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ))  :  
                                                add_filech4 = CH4Files[m] 
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
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    for m in range(len(CH4Times)):
        
                                        mseconds = abs(np.int(CH4StartTime[m])) % 100
                                        mminutes = ((abs(np.int(CH4StartTime[m])) % 10000) - mseconds)/100
                                        mhour = (np.int(CH4StartTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                                        if(delta != np.int(9)):
                                            if(np.int(inputCH4EndTime)%10 != 9): 
                                                timediff = 0 
                                            else: 
                                                timediff = 1 
                                        else:
                                            timediff = 1
        
                                        if( (np.int(hseconds) == 59   ) and (np.int(hminutes) == 59   )  ): # change hour and minute and second of m file 
                                            if( ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(hhour + 1)   )) or  ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(hhour )   ))    ):  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                        elif( (np.int(hseconds) == 59   ) and (np.int(hminutes) != 59   )  ): #  minute and second of m file V
                                            if( ((np.int(mseconds) == 00   ) and (np.int(mminutes) == (np.int(hminutes) + 1)   ) and (np.int(mhour) == np.int(hhour )   ) ) or ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(hminutes) )   ) and (np.int(mhour) == np.int(hhour )   ) )    ):  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                        elif( (np.int(hseconds) != 59   ) and (np.int(hminutes) == 59   )  ): #  minute and second of m file V
                                            if( math.isclose(np.int(mseconds) ,(np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ))  :  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                        elif( (np.int(hseconds) != 59   ) and (np.int(hminutes) != 59   )  ): #  minute and second of m file V
                                            if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                add_filech4 = CH4Files[m] 
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
                                        
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                        # find previous ch4 granule last #|ashift| frames... 
                                        for m in range(len(CH4Times)):
        
                                            mseconds = abs(np.int(CH4EndTime[m])) % 100
                                            mminutes = ((abs(np.int(CH4EndTime[m])) % 10000) - mseconds)/100
                                            mhour = (np.int(CH4EndTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                                            if(delta != np.int(9)):
                                                if(np.int(inputCH4StartTime)%10 != 0): 
                                                    timediff = 0 
                                                else: 
                                                    timediff = 1 
        
                                            else:
                                                timediff = 1
                                            if( (np.int(hseconds) == 0   ) and (np.int(hminutes) == 0   )  ): # change hour and minute and second of m file 
                                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(hhour - 1)   )) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(hhour )   ))    ):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                            elif( (np.int(hseconds) == 0   ) and (np.int(hminutes) != 0   )  ): #  minute and second of m file V
                                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(hminutes) -1)   ) and (np.int(mhour) == np.int(hhour )   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == (np.int(hminutes) )   ) and (np.int(mhour) == np.int(hhour )   ) ) ):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                            elif( (np.int(hseconds) != 0   ) and (np.int(hminutes) == 0   )  ): #  minute and second of m file V
                                                if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                            elif( (np.int(hseconds) != 0   ) and (np.int(hminutes) != 0   )  ): #  minute and second of m file V
                                                if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   )) :  
                                                    add_filech4 = CH4Files[m] 
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
                                        
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                        # find previous ch4 granule last #|ashift| frames... 
                                        for m in range(len(CH4Times)):
        
                                            mseconds = abs(np.int(CH4EndTime[m])) % 100
                                            mminutes = ((abs(np.int(CH4EndTime[m])) % 10000) - mseconds)/100
                                            mhour = (np.int(CH4EndTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                                            if(delta != np.int(9)):
                                                if(np.int(inputCH4StartTime)%10 != 0): 
                                                    timediff = 0 
                                                else: 
                                                    timediff = 1 
        
                                            else:
                                                timediff = 1
                                            if( (np.int(hseconds) == 0   ) and (np.int(hminutes) == 0   )  ): # change hour and minute and second of m file 
                                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(hhour - 1)   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(hhour )   ) )):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                            elif( (np.int(hseconds) == 0   ) and (np.int(hminutes) != 0   )  ): #  minute and second of m file V
                                                if(( (np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(hminutes) -1)   ) and (np.int(mhour) == np.int(hhour )   ) ) or ( (np.int(mseconds) == 0   ) and (np.int(mminutes) == (np.int(hminutes) )   ) and (np.int(mhour) == np.int(hhour )   ) )):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                            elif( (np.int(hseconds) != 0   ) and (np.int(hminutes) == 0   )  ): #  minute and second of m file V
                                                if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                            elif( (np.int(hseconds) != 0   ) and (np.int(hminutes) != 0   )  ): #  minute and second of m file V
                        
                                                if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
        
                                        for m in range(len(CH4Times)):
        
                                            mseconds = abs(np.int(CH4StartTime[m])) % 100
                                            mminutes = ((abs(np.int(CH4StartTime[m])) % 10000) - mseconds)/100
                                            mhour = (np.int(CH4StartTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                                            if(delta != np.int(9)):
                                                if(np.int(inputCH4EndTime)%10 != 9): 
                                                    timediff = 0 
                                                else: 
                                                    timediff = 1 
                                            else:
                                                timediff = 1
                                            if( (np.int(hseconds) == 59   ) and (np.int(hminutes) == 59   )  ): # change hour and minute and second of m file 
                                                if( ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(hhour + 1)   ) ) or  ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(hhour )   ) )):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                            elif( (np.int(hseconds) == 59   ) and (np.int(hminutes) != 59   )  ): #  minute and second of m file V
                                                if( ((np.int(mseconds) == 00   ) and (np.int(mminutes) == (np.int(hminutes) + 1)   ) and (np.int(mhour) == np.int(hhour )   ) ) or((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(hminutes) )   ) and (np.int(mhour) == np.int(hhour )   ) )):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                        
                                            elif( (np.int(hseconds) != 59   ) and (np.int(hminutes) == 59   )  ): #  minute and second of m file V
                                                if( math.isclose(np.int(mseconds) ,  (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes) )   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                    
                                                    add_filech4 = CH4Files[m] 
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
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                            elif( (np.int(hseconds) != 59   ) and (np.int(hminutes) != 59   )  ): #  minute and second of m file V
                                                if( math.isclose(np.int(mseconds) == (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
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
        
                                        rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                        for m in range(len(CH4Times)):
        
                                            mseconds = abs(np.int(CH4EndTime[m])) % 100
                                            mminutes = ((abs(np.int(CH4EndTime[m])) % 10000) - mseconds)/100
                                            mhour = (np.int(CH4EndTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                                            if(delta != np.int(9)):
                                                if(np.int(inputCH4StartTime)%10 != 0): 
                                                    timediff = 0 
                                                else: 
                                                    timediff = 1 
        
                                            else:
                                                timediff = 1
                                            if( (np.int(hseconds) == 0   ) and (np.int(hminutes) == 0   )  ): # change hour and minute and second of m file 
                                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(hhour - 1)   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(hhour )   ) )):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                            elif( (np.int(hseconds) == 0   ) and (np.int(hminutes) != 0   )  ): #  minute and second of m file V
                                                if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(hminutes) -1)   ) and (np.int(mhour) == np.int(hhour )   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == (np.int(hminutes) )   ) and (np.int(mhour) == np.int(hhour )   ) )):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                            elif( (np.int(hseconds) != 0   ) and (np.int(hminutes) == 0   )  ): #  minute and second of m file V
                                                if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                    add_filech4 = CH4Files[m] 
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
                                                    
                                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                            elif( (np.int(hseconds) != 0   ) and (np.int(hminutes) != 0   )  ): #  minute and second of m file V
                                                if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                    add_filech4 = CH4Files[m] 
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
                                    
                                    rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                    for m in range(len(CH4Times)):
        
                                        mseconds = abs(np.int(CH4EndTime[m])) % 100
                                        mminutes = ((abs(np.int(CH4EndTime[m])) % 10000) - mseconds)/100
                                        mhour = (np.int(CH4EndTime[m]) - mseconds - mminutes*100 ) / 10000 
        
                                        if(delta != np.int(9)):
                                            if(np.int(inputCH4StartTime)%10 != 0): 
                                                timediff = 0 
                                            else: 
                                                timediff = 1 
        
                                        else:
                                            timediff = 1
                                        if( (np.int(hseconds) == 0   ) and (np.int(hminutes) == 0   )  ): # change hour and minute and second of m file 
                                            if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == 59   ) and (np.int(mhour) == np.int(hhour - 1)   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == 0   ) and (np.int(mhour) == np.int(hhour )   ) )):  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                        elif( (np.int(hseconds) == 0   ) and (np.int(hminutes) != 0   )  ): #  minute and second of m file V
                                            if( ((np.int(mseconds) == 59   ) and (np.int(mminutes) == (np.int(hminutes) -1)   ) and (np.int(mhour) == np.int(hhour )   ) ) or  ((np.int(mseconds) == 0   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) )):  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
                                        elif( (np.int(hseconds) != 0   ) and (np.int(hminutes) == 0   )  ): #  minute and second of m file V
                                            if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                add_filech4 = CH4Files[m] 
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
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                        elif( (np.int(hseconds) != 0   ) and (np.int(hminutes) != 0   )  ): #  minute and second of m file V
                                            if( math.isclose(np.int(mseconds) , (np.int(hseconds)),abs_tol=1   ) and (np.int(mminutes) == (np.int(hminutes))   ) and (np.int(mhour) == np.int(hhour )   ) ):  
                                                add_filech4 = CH4Files[m] 
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
                                                
                                                rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new = self.assign_data(rad_new,rad_err_new,wvl_new,rad_flags_new,lon_new,lat_new,corner_lon_new,corner_lat_new,vza_new,sza_new,vaa_new,saa_new,aza_new,obsalt_new,surfalt_new,time_new,ac_lon_new,ac_lat_new,ac_alt_surf_new,ac_surf_alt_new,ac_pix_bore_new,ac_pos_new,a1,a2,b1,b2,c1,c2,d1,d2)
        
                                    
                                    
        
                                else:
                                    print("Missed Something!")
                                    exit()
                
        
                                #############################
                                #  NANS APPLIED TO CH4 DATA FROM O2 DATA POSSESSING NANS 
                                #############################
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
        
                                #valpix = np.isfinite(y2['CornerLongitude'][:,:,:])
                                
                                #corner_lon_new[valpix!=True] = np.nan
                                #corner_lat_new[valpix!=True] = np.nan
        
                                #############################
                                #############################
        
                                #valpix = np.isfinite(y2['Longitude'][:,:])
        
                                #lon_new[valpix!=True] = np.nan
                                #lat_new[valpix!=True] = np.nan
                                #sza_new[valpix!=True] = np.nan
                                #saa_new[valpix!=True] = np.nan
                                #vza_new[valpix!=True] = np.nan
                                #vaa_new[valpix!=True] = np.nan
                                #saa_new[valpix!=True] = np.nan
                                #aza_new[valpix!=True] = np.nan
                                #surfalt_new[valpix!=True] = np.nan 
                                #valpix3d = np.repeat(valpix[np.newaxis,...],3,axis=0)
                                #ac_pix_bore_new[valpix3d!=True] = np.nan
        
                                #############################
                                #############################
                                #valpix = np.isfinite(y2['Time'][:])
        
                                #obsalt_new[valpix!=True] = np.nan
                                #time_new[valpix!=True] = np.nan
                                #ac_alt_surf_new[valpix!=True] = np.nan
                                #ac_surf_alt_new[valpix!=True] = np.nan
                                #ac_lon_new[valpix!=True] = np.nan
                                #ac_lat_new[valpix!=True] = np.nan
        
                                #valpixel2d = np.repeat(valpix[np.newaxis,...],3,axis=0)
                                #ac_pos_new[valpixel2d!=True] = np.nan
        
                                #datao2[valpix!=True] = np.nan
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
                                
                                #corner_lon_o2[valpix!=True] = np.nan
                                #corner_lat_o2[valpix!=True] = np.nan
        
                                #valpix = np.isfinite(lon_new)
        
                                lon_o2 = y2['Longitude'][:,:] 
                                lat_o2 = y2['Latitude'][:,:] 
                                sza_o2 = y2['SolarZenithAngle'][:,:] 
                                saa_o2 = y2['SolarAzimuthAngle'][:,:] 
                                aza_o2 = y2['RelativeAzimuthAngle'][:,:] 
                                vza_o2 = y2['ViewingZenithAngle'][:,:] 
                                vaa_o2 = y2['ViewingAzimuthAngle'][:,:] 
                                surfalt_o2 = y2['SurfaceAltitude'][:,:] 
                                ac_lon_o2 = y2['Aircraft_Longitude'][:]
                                ac_lat_o2 = y2['Aircraft_Latitude'][:]
                                ac_alt_surf_o2 = y2['Aircraft_AltitudeAboveSurface'][:]
                                ac_surf_alt_o2 = y2['Aircraft_SurfaceAltitude'][:]
                                ac_pix_bore_o2 = y2['Aircraft_PixelBore'][:,:,:]
                                ac_pos_o2 = y2['Aircraft_Pos'][:,:]
        
                                #lon_o2[valpix!=True] = np.nan
                                #lat_o2[valpix!=True] = np.nan
                                #sza_o2[valpix!=True] = np.nan
                                #saa_o2[valpix!=True] = np.nan
                                #vza_o2[valpix!=True] = np.nan
                                #vaa_o2[valpix!=True] = np.nan
                                #saa_o2[valpix!=True] = np.nan
                                #aza_o2[valpix!=True] = np.nan
                                #surfalt_o2[valpix!=True] = np.nan
          
                                #valpix3d = np.repeat(valpix[np.newaxis,...],3,axis=0)
                                #ac_pix_bore_o2[valpix3d!=True] = np.nan
        
                                #############################
                                #############################
                                valpix = np.isfinite(time_new)
        
                                obsalt_o2 = y2['ObservationAltitude'][:] 
                                time_o2 = y2['Time'][:] 
        
                                #obsalt_o2[valpix!=True] = np.nan
                                #time_o2[valpix!=True] = np.nan
                                #ac_alt_surf_o2[valpix!=True] = np.nan
                                #ac_surf_alt_o2[valpix!=True] = np.nan
                                #ac_lon_o2[valpix!=True] = np.nan 
                                #ac_lat_o2[valpix!=True] = np.nan 
        
                                #valpixel2d = np.repeat(valpix[np.newaxis,...],3,axis=0)
                                #ac_pos_o2[valpixel2d!=True] = np.nan
        
                                #datao2[valpix!=True] = np.nan
        
        
        
                                #############################
                                # NOW WE WRITE THE NEW FILE TO DESK: CH4 FIRST
                                #############################
                                xtrk_aggfac = 15
                                atrk_aggfac = 3
                                filename = filech4.split(".nc")[0]
                                filename = filename.split("CH4_NATIVE/")[1]
                                l1b_ch4_dir = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02/EKC_V4_with_Stray/CH4_15x3_Aligned/' 
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
                                
                                l1 = pysplat.level1(l1_outfile,lon_new,lat_new,obsalt_new,time_new,ac_lon_new,ac_lat_new,ac_pos_new,ac_surf_alt_new,ac_alt_surf_new,ac_pix_bore_new,optbenchT=None,clon=corner_lon_new,clat=corner_lat_new)
                                l1.set_2d_geofield('SurfaceAltitude', surfalt_new)
                                l1.set_2d_geofield('SolarZenithAngle', sza_new)
                                l1.set_2d_geofield('SolarAzimuthAngle', saa_new)
                                l1.set_2d_geofield('ViewingZenithAngle', vza_new)
                                l1.set_2d_geofield('ViewingAzimuthAngle', vaa_new)
                                l1.set_2d_geofield('RelativeAzimuthAngle', aza_new)
                                l1.add_radiance_band(wvl_new,rad_new,rad_err=rad_err_new,rad_flag=rad_flags_new)
                                l1.close()
        
                                  
                                  
                                found_match=False
                                for abc in range(len(native_ch4_files)):
                                    if(found_match==False and (str(os.path.basename(filech4) == str(native_ch4_files[abc]))) ):
                                        priority =native_ch4_priority[abc]      
                                        found_match=True
        
                                lockname=logfile_ch4+'.lock'
                                with FileLock(lockname):
                                    f = open(logfile_ch4,'a+') 
                                    f.write(str(l1_outfile)+' '+str(priority)+'\n' )
                                    f.close()
        
        
        
                                #############################
                                # NOW WE WRITE THE NEW FILE TO DESK: O2 SECOND
                                #############################
        
                                filename = fileo2.split(".nc")[0]
                                filename = filename.split("O2_NATIVE/")[1]
                                l1b_o2_dir = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02/EKC_V4_with_Stray/O2_15x3_Aligned/' 
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
                                
                                l1 = pysplat.level1(l1_outfile,lon_o2,lat_o2,obsalt_o2,time_o2,ac_lon_o2,ac_lat_o2,ac_pos_o2,ac_surf_alt_o2,ac_alt_surf_o2,ac_pix_bore_o2,optbenchT=None,clon=corner_lon_o2,clat=corner_lat_o2)
                                l1.set_2d_geofield('SurfaceAltitude', surfalt_o2)
                                l1.set_2d_geofield('SolarZenithAngle', sza_o2)
                                l1.set_2d_geofield('SolarAzimuthAngle', saa_o2)
                                l1.set_2d_geofield('ViewingZenithAngle', vza_o2)
                                l1.set_2d_geofield('ViewingAzimuthAngle', vaa_o2)
                                l1.set_2d_geofield('RelativeAzimuthAngle', aza_o2)
                                l1.add_radiance_band(wvl_o2,rad_o2,rad_err=rad_err_o2,rad_flag=rad_flags_o2)
                                l1.close()
                                  
        
                                lockname=logfile_o2+'.lock'
                                with FileLock(lockname):
                                    f = open(logfile_o2,'a+') 
                                    f.write(str(l1_outfile)+' '+str(priority)+'\n' )
                                    f.close()
                                  
                                  
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
        
                                filename = os.path.join('Final_'+inputCH4Times+'.png')
                                plt.savefig(filename,dpi=1000)
                                plt.close()

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
        surfalt[a1:a2] = self.x2['SurfaceAltitude'][c1:c2]
        obsalt[a1:a2] = self.x2['ObservationAltitude'][c1:c2]
        time[a1:a2] = self.x2['Time'][c1:c2]
        ac_lon[a1:a2] = self.x2['Aircraft_Longitude'][c1:c2]
        ac_lat[a1:a2] = self.x2['Aircraft_Latitude'][c1:c2]
        ac_alt_surf[a1:a2] = self.x2['Aircraft_AltitudeAboveSurface'][c1:c2]
        ac_surf_alt[a1:a2] = self.x2['Aircraft_SurfaceAltitude'][c1:c2]
        ac_pix_bore[:,a1:a2,b1:b2] = self.x2['Aircraft_PixelBore'][:,c1:c2,d1:d2]
        ac_pos[:,a1:a2] = self.x2['Aircraft_Pos'][:,c1:c2]
    
        for i in range(a1,a2):
            for j in range(b1,b2):
                aza[i,j] = vaa[i,j] - (180.0 - saa[i,j])
                if( aza[i,j] < 0.0 ):
                    aza[i,j] = aza[i,j] + 360.0
                if( aza[i,j] > 360.0 ):
                    aza[i,j] = aza[i,j] - 360.0
    
        return (rad,raderr,wvl,radflag,lon,lat,clon,clat,vza,sza,vaa,saa,aza,obsalt,surfalt,time,ac_lon,ac_lat,ac_alt_surf,ac_surf_alt,ac_pix_bore,ac_pos)
