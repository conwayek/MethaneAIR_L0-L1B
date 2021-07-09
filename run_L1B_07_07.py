"""
Created on Thu Sep 29 14:25:00 2020# -*- coding: utf-8 -*-
@authors: kangsun & eamonconway
"""
from memory_profiler import profile
import numpy as np
import sys, os
import math
import logging
import datetime as dt
from netCDF4 import Dataset
from scipy import interpolate,optimize
from matplotlib import pyplot as plt
from sys import exit
#os.environ['R_HOME'] = "/usr/local/Cellar/r/4.0.2_1/lib/R"
import csv
import time

import subprocess
import akaze_nc_ch4_o2_06_24_2021
import akaze_nc_06_24_2021
import aggregate
import dem_maker 

t_start = time.time()

flight = 'RF02'
computer = 'Odyssey' # or Odyssey
molecule = 'o2'

if(computer == 'Odyssey'):
    import rpy2 as rpy2
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, packages

# This is the date of the flight

if(flight == 'RF01'):
    datenow = dt.datetime(year = 2019, month = 11, day = 08)
elif(flight == 'RF02'):
    datenow = dt.datetime(year = 2019, month = 11, day = 12)



if(computer == 'Odyssey'):
    root_dest = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/'
    root_data = '/n/holylfs04/LABS/wofsy_lab/Lab/econway/DATA/'
    if(flight == 'RF01'):
        ac_file = 'MethaneAIRrf01_hrt.nc' 
        if(molecule == 'ch4'):
            root_l0 = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/data/flight_testing/20191108/ch4_camera/'
            seq_files = os.path.join(root_data,'files.ch4.rf01.txt')
        elif(molecule == 'o2'):
            root_l0 = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/data/flight_testing/20191108/o2_camera/'
            seq_files = os.path.join(root_data,'files.o2.rf01.txt')
    elif(flight == 'RF02'):
        ac_file = 'MethaneAIRrf02_hrt.nc' 
        if(molecule == 'ch4'):
            root_l0 = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/data/flight_testing/20191112/ch4_camera/'
            seq_files = os.path.join(root_data,'ch4_seq_files_RF02.txt')
        elif(molecule == 'o2'):
            root_l0 = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/data/flight_testing/20191112/o2_camera/'
            seq_files = os.path.join(root_data,'o2_seq_files_RF02.txt')
    else:
        print('Bad Research Flight')
        exit()

elif(computer == 'Hydra'):
    root_dest = '/scratch/sao/econway/L1B_Files/'
    root_data = '/home/econway/DATA/'
    if(flight == 'RF01'):
        ac_file = 'MethaneAIRrf01_hrt.nc' 
        if(molecule == 'ch4'):
            root_l0 = '/scratch/sao/econway/MethaneAIR_L0_RF01_CH4/'
            seq_files = os.path.join(root_data,'files.ch4.rf01.txt')
        elif(molecule == 'o2'):
            root_l0 = '/scratch/sao/econway/MethaneAIR_L0_RF01_O2/'
            seq_files = os.path.join(root_data,'files.o2.rf01.txt')
    elif(flight == 'RF02'):
        ac_file = 'MethaneAIRrf02_hrt.nc' 
        if(molecule == 'ch4'):
            root_l0 = '/scratch/sao/econway/MethaneAIR_L0_RF02_CH4/'
            seq_files = os.path.join(root_data,'ch4_seq_files_RF02.txt')
        elif(molecule == 'o2'):
            root_l0 = '/scratch/sao/econway/MethaneAIR_L0_RF02_O2/'
            seq_files = os.path.join(root_data,'o2_seq_files_RF02.txt')
    else:
        print('Bad Research Flight')
        exit()




l1_abs_akaze_DataDir = os.path.join(root_dest,'RF02/EKC_V4/O2_ABS_AKAZE_FILES/')
l1_rel_akaze_DataDir = os.path.join(root_dest,'RF02/EKC_V4/CH4_REL_AKAZE_FILES/')

if(molecule == 'ch4'):
    l1DataDir = os.path.join(root_dest,str(flight)+'/EKC_V4/CH4_NATIVE/')
    l1AggDataDir = os.path.join(root_dest,str(flight)+'/EKC_V4/CH4_15x3/')
if(molecule == 'o2'):
    l1DataDir = os.path.join(root_dest,str(flight)+'/EKC_V4/O2_NATIVE/')
    l1AggDataDir = os.path.join(root_dest,str(flight)+'/EKC_V4/O2_15x3/')

windowTransmissionPath = os.path.join(root_data,'window_transmission.mat')
# CH4 CHANNEL DATA    
badPixelMapPathCH4 = os.path.join(root_data,'CH4_bad_pix.csv')
radCalPathCH4 = os.path.join(root_data,'rad_coef_ch4_100ms_ord4_20200209T211211.mat')
wavCalPathCH4 = os.path.join(root_data,'methaneair_ch4_spectroscopic_calibration_1280_footprints.nc')
l0DataDirCH4 = root_l0
strayLightKernelPathCH4 = os.path.join(root_data,'K_stable_CH4.mat')
solarPathCH4= os.path.join(root_data,'hybrid_reference_spectrum_p001nm_resolution_with_unc.nc')
pixellimitCH4=[200,620]
pixellimitXCH4=[100,1000]
#pixellimitXCH4=[100,1000]
spectraReffileCH4=os.path.join(root_data,'spectra_CH4_channel.nc')
refitwavelengthCH4 = True
calibratewavelengthCH4 = True
fitisrfCH4 = False
isrftypeCH4 = 'ISRF' 
fitSZACH4 = True

# O2 CHANNEL DATA    
badPixelMapPathO2 = os.path.join(root_data,'O2_bad_pix.csv')
radCalPathO2 = os.path.join(root_data,'rad_coef_o2_100ms_ord4_20200205T230528.mat')
wavCalPathO2 = os.path.join(root_data,'methaneair_o2_spectroscopic_calibration_1280_footprints.nc')
l0DataDirO2 = root_l0
strayLightKernelPathO2 = os.path.join(root_data,'K_stable_O2.mat')
solarPathO2= os.path.join(root_data,'hybrid_reference_spectrum_p001nm_resolution_with_unc.nc')
pixellimitO2=[500,620]
pixellimitXO2=[100,1000]
spectraReffileO2=os.path.join(root_data,'spectra_O2_channel.nc')
refitwavelengthO2 = True
calibratewavelengthO2 = True
fitisrfO2 = False 
isrftypeO2 = 'ISRF'
fitSZAO2 = True

calidatadir = root_data 

xtol = 1e-2
ftol = 1e-2
xfac =15
yfac = 3 





#READ IN ALL THE FILE NAMES FROM THE OBSERVATION
with open(os.path.join(root_data,seq_files), 'r') as file:
	filenames = file.read().splitlines()
file.close()


# THIS IS NUMBER THE FILE WE ARE GOING TO PROCESS
for line in open('iteration.txt', 'r'):
  values = [float(s) for s in line.split()]
  iteration = np.int(values[0])


targetfile = filenames[iteration]

#####
# HERE WE GET THE TIME FROM THE GRANULES AND GET THEIR ALTITUDE: ACCURATE TO 100-200 METER OR SO ON AS(DE)CENDING LEGS OF JOURNEY
#####
ncid = Dataset(os.path.join(root_data,ac_file), 'r')

altitude1 = ncid.variables['GGALT'][:]
altitude2 = ncid.variables['GGEOIDHT'][:]
timeGPS = ncid.variables['GTIME_A'][:]
solze = ncid.variables['SOLZE'][:,0]

height = np.zeros(len(timeGPS))
height = altitude1 + altitude2

index = np.zeros(len(timeGPS))
for i in range(len(timeGPS)):
	index[i] = i 

x = interpolate.interp1d(index,height) 
SZA_interp = interpolate.interp1d(index,solze) 


#GET ALL THE TIME STAMPS
hour = np.zeros(len(filenames))
minutes = np.zeros(len(filenames))
seconds = np.zeros(len(filenames))
timetotal = np.zeros(len(filenames))





for i in range(len(filenames)):
    temp = filenames[i].split('_camera_')[1]
    temp = dt.datetime.strptime(temp.strip('.seq'),'%Y_%m_%d_%H_%M_%S')
    hour[i] = temp.hour
    minutes[i] = temp.minute
    seconds[i] = temp.second
    timetotal[i] = hour[i]*60.0*60.0 + minutes[i]*60.0 + seconds[i]


timestart = timetotal[0]
timeend = timetotal[-1]

flighttime = np.int(timeend - timestart)


#need to convert the number of measurements, into number of points in index
scale = len(index)/len(timetotal)


indexfl = np.zeros(len(timetotal))
for i in range(len(timetotal)):
	indexfl[i] = i*scale



altitude_it = np.zeros(len(timetotal))
SZA_it = np.zeros(len(timetotal))

for i in range(len(timetotal)):
    altitude_it[i] = x(indexfl[i])
    SZA_it[i] = SZA_interp(indexfl[i])

SZA = SZA_it[iteration]*3.14159/180.0

altitude = altitude_it[iteration]/1000.0

if(SZA < 0 or altitude < 0 ):
    SZA = SZA_it[iteration+1]*3.14159/180.0
    altitude = altitude_it[iteration+1]/1000.0
    


if(flight == 'RF01'):
    if(altitude[iteration] <= (25000*0.3048)  ):
        if(molecule == 'ch4'):
    	    darkfile = os.path.join(root_l0,'ch4_camera_2019_11_08_17_42_53.seq') 
        elif(molecule == 'o2'):
    	    darkfile = os.path.join(root_l0,'o2_camera_2019_11_08_17_42_53.seq') 
    else:
    	#CH4
        if(molecule == 'ch4'):
    	    darkfile = os.path.join(root_l0,'ch4_camera_2019_11_08_18_25_38.seq') 
    	#O2
        if(molecule == 'o2'):
    	    darkfile = os.path.join(root_l0,'o2_camera_2019_11_08_18_25_39.seq') 

elif(flight == 'RF02'):

    if(iteration <= 216):
        if(molecule == 'ch4'):
            darkfile = os.path.join(root_l0,'ch4_camera_2019_11_12_17_53_15.seq')
        elif(molecule == 'o2'):
            darkfile = os.path.join(root_l0,'o2_camera_2019_11_12_17_53_15.seq')
    else:
        if(molecule == 'ch4'):
            darkfile = os.path.join(root_l0,'ch4_camera_2019_11_12_20_43_58.seq')
        if(molecule == 'o2'):
            darkfile = os.path.join(root_l0,'o2_camera_2019_11_12_20_43_58.seq')

elif(flight == 'RF03'):
    import darkfinder
    darkfiles = darkfinder.main(root_l0)
    
    # get the current L0 filetime 
    temp = targetfile.split('_camera_')[1]
    time_now = dt.datetime.strptime(temp.strip('.seq'),'%Y_%m_%d_%H_%M_%S')    
    
    for i in range(len(darkfiles)):
        continu = True
        t = (darkfiles[i]).split('_camera_')[1]
        t = dt.datetime.strptime(temp.strip('.seq'),'%Y_%m_%d_%H_%M_%S')
        if( ( t < time_now ) and (continu == True)  ):
           coninu = True 
        else:
           coninu = False 
           darkfile = darkfiles[i-1]



#########################################################################################
#########################################################################################


from MethaneAIR_L1_EKC_07_08_2021 import MethaneAIR_L1

if(molecule == 'ch4':)
    #########################################################################################
    #########################################################################################
    # IF RF02 SET THESE TO ZERO
    # test dark and one native granule for CH4 camera
    m = MethaneAIR_L1(whichBand='CH4',
    l0DataDir=l0DataDirCH4,
    l1DataDir=l1DataDir,
    badPixelMapPath=badPixelMapPathCH4,
    radCalPath=radCalPathCH4,
    wavCalPath=wavCalPathCH4,
    windowTransmissionPath=windowTransmissionPath,
    solarPath=solarPathCH4,
    pixellimit=pixellimitCH4,
    pixellimitX=pixellimitXCH4,
    spectraReffile=spectraReffileCH4,
    refitwavelength=refitwavelengthCH4,
    calibratewavelength=calibratewavelengthCH4,
    ISRFtype=isrftypeCH4,
    fitisrf=fitisrfCH4,
    calidata = calidatadir,
    fitSZA = fitSZACH4,
    SolZA = SZA,
    ALT = altitude,
    xtol=xtol,
    ftol=ftol,
    xtrackaggfactor=xfac,
    atrackaggfactor=yfac)
                      
    
    #STRAY LIGHT CORRECTION         
    #m.F_stray_light_input(strayLightKernelPath=strayLightKernelPathCH4,rowExtent=200,colExtent=300)
    
    
    
    
    dark = m.F_grab_dark(darkfile)
    #########################################################################################
        
    
    # SET CURRENT WORKING DIRECTORY AND OUTPUT LOCATION
    cwd = os.getcwd()
    
    if not os.path.exists(os.path.join(cwd,'0Outputs')):
        os.makedirs(os.path.join(cwd,'0Outputs'))
    
    savepath = os.path.join(cwd,'0Outputs')
    
    #########################################################################################
    #PROCESS GRANULE WITH DARK DATA SUBTRACTED
    granule = m.F_granule_processor(os.path.join(l0DataDirCH4,targetfile),dark,ePerDN=4.6)
    
    
    timestamp = np.min(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                              +np.max(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                              +dt.datetime.now().strftime('%Y%m%dT%H%M%S')

    if(flight == 'RF01'):
        flight_nc_file = '0Inputs/MethaneAIRrf01_hrt.nc'
        eop_file = '0Inputs/eop19622020.txt'
    elif(flight == 'RF02'):
        flight_nc_file = '0Inputs/MethaneAIRrf02_hrt.nc'
        eop_file = '0Inputs/eop19622020.txt'

    mintime = np.min(granule.frameDateTime)
    maxtime = np.max(granule.frameDateTime)

    FOV=33.7
    buff = 15000
    dem_file = dem_maker(os.path.join(root_data,flight_nc_file),mintime,maxtime,datenow,FOV,buff) 

    os.system("gdal_translate -of GTiff dem.nc dem.tiff")

    filename = os.path.join(cwd,'MethaneAIR_L1B_CH4_'+timestamp+'.nc')
    #########################################################################################
    # SAVE ENTIRE .NC GRANULE
    m.F_save_L1B_mat(timestamp,granule,headerStr='MethaneAIR_L1B_CH4_')
    #########################################################################################
    #LOAD GEOLOCATION FUNCTION IN R
    # If computer is Hydra
    if(computer == 'Hydra'):
        args1=str(os.path.join(root_data,flight_nc_file))
        args2=str('dem.tiff')
        args3=str(filename)
        args4=str(savepath)
        args5=str(os.path.join(root_data,'0Inputs/eop19622020.txt'))
        args6=str(os.path.join(root_data,'0User_Functions/'))
        subprocess.call(['/home/econway/anaconda3/envs/r4-base/bin/Rscript', 'Orthorectification_Avionics_NC_Fast_Hydra.R', args1, args2, args3, args4, args5, args6])
    else:
        robjects.r('''source('Orthorectification_Avionics_NC_Fast.R')''')
        r_getname = robjects.globalenv['Orthorectification_Avionics_NC']
        # Avionics Only Routine
        r_getname(file_flightnc = os.path.join(root_data,flight_nc_file),\
        file_dem = 'dem.tiff',\
        file_L1=filename,\
        dir_output = savepath,\
        latvar = "LATC",\
        lonvar = "LONC",\
        file_eop = os.path.join(root_data,eop_file),\
        dir_lib = os.path.join(root_data,'0User_Functions/'),\
        FOV = 33.7)
    #########################################################################################
    # WRITE SPLAT FORMATTED L1B FILE - ENTIRE GRANULE 
    # need to save the avionics only full-granule to a temporary scratch directpry for ch4 cross-alignment
    if not os.path.exists(os.path.join(cwd,'AvionicsGranule')):
        os.makedirs(os.path.join(cwd,'AvionicsGranule'))
    avionics_dir = os.path.join(cwd,'AvionicsGranule')
    x = m.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(avionics_dir),xtrk_aggfac=1,atrk_aggfac=1,ortho_step='avionics')
    os.remove(os.path.join(cwd,'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'))
    os.remove(os.path.join(savepath,'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'))
    #########################################################################################
    # NEED TO CALL AKAZE ALGORITHM HERE - NEED RELATIVE ADJUSTMENT PARAMETERS
    #########################################################################################
    # Current CH4 granule times
    t = timestamp.split("_")
    ch4_tzero = t[0]
    ch4_tend = t[1]
    ch4_year = ch4_tzero[0:4]
    ch4_month = ch4_tzero[4:6]
    ch4_day = ch4_tzero[6:8]
    ch4_hour = ch4_tzero[9:11]
    ch4_minute = ch4_tzero[11:13]
    ch4_second = int(ch4_tzero[13:15])
    
    
    o2dir = '/n/holyscratch01/wofsy_lab/econway/RF02_Post_Corrections/O2_NATIVE/'
    o2file=[]
    for folder in os.listdir(o2dir):
        if(folder.endswith('seq')):
            f = folder.split(".seq")[0]
            f = f.split("o2_camera_")[1]
            f = f.split("_")
            fyear = f[0]
            fmonth = f[1]
            fday = f[2]
            fhour = f[3]
            fminute = f[4]
            fsecond = int(f[5])
    
            if(( fyear == ch4_year) and (fminute==ch4_minute)and(fhour==ch4_hour)and( math.isclose(fsecond,ch4_second,abs_tol=1.1)) and(fday==ch4_day)and(fmonth==ch4_month)):
               filedir = os.path.join(o2dir+folder+'/AvionicsGranule/')
               for file in os.listdir(filedir):
                   if(file.endswith('.nc')):
                       o2file.append(os.path.join(filedir,str(file).split('.nc')[0]))
    
    ch4file = []
    ch4file.append(os.path.join(avionics_dir,'MethaneAIR_L1B_CH4_'+timestamp))
    if(o2file == []):
        print('No Matched O2 File')
        exit()
    rel_sc_lat,rel_sc_lon,rel_off_lat,rel_off_lon,rval_lat,rval_lon = akaze_nc_ch4_o2_06_24_2021.main(o2file, ch4file)
    rel_off_lon = np.float(rel_off_lon) 
    rel_sc_lon  = np.float(rel_sc_lon ) 
    rel_off_lat = np.float(rel_off_lat) 
    rel_sc_lat  = np.float(rel_sc_lat )
    
    #########################################################################################
    # NEED TO FIND THE RESPECTIVE ABSOLUTE SHIFTS IN THE O2 DIRECTORIES
    #########################################################################################
    # Here we get all the akaze results for O2 and store them.
    list_abs_files = []
    tzero_stamp_hr = []
    tzero_stamp_min = []
    tzero_stamp_sec = []
    tend_stamp_hr = []
    tend_stamp_min = []
    tend_stamp_sec = []
    tzero_stamp = []
    tend_stamp = []
    tproc_stamp = []
    for file in os.listdir(l1_abs_akaze_DataDir):
        if(file.endswith('akaze.txt')):
            list_abs_files.append(file)
            t=file.split("_absolute_correction_akaze.txt")[0]
            t=t.split("MethaneAIR_L1B_O2_")[1]
            t=t.split("_")
            tzero_stamp_hr.append((dt.datetime.strptime(t[0],"%Y%m%dT%H%M%S")).hour)
            tzero_stamp_min.append((dt.datetime.strptime(t[0],"%Y%m%dT%H%M%S")).minute)
            tzero_stamp_sec.append((dt.datetime.strptime(t[0],"%Y%m%dT%H%M%S")).second)
            tend_stamp_hr.append((dt.datetime.strptime(t[1],"%Y%m%dT%H%M%S")).hour)
            tend_stamp_min.append((dt.datetime.strptime(t[1],"%Y%m%dT%H%M%S")).minute)
            tend_stamp_sec.append((dt.datetime.strptime(t[1],"%Y%m%dT%H%M%S")).second)
            tzero_stamp.append(t[0])
            tend_stamp.append(t[1])
            tproc_stamp.append(t[2])
    #########################################################################################
    
    t = timestamp.split("_")
    nfiles_o2 = len(list_abs_files) 
    done = False
    for i in range(nfiles_o2):
        if( ( ( (dt.datetime.strptime(timestamp.split("_")[0],"%Y%m%dT%H%M%S")).hour ) == tzero_stamp_hr[i])  and ( ( (dt.datetime.strptime(timestamp.split("_")[0],"%Y%m%dT%H%M%S")).minute ) == tzero_stamp_min[i]) 
        and (math.isclose( ( (dt.datetime.strptime(timestamp.split("_")[0],"%Y%m%dT%H%M%S")).second ),tzero_stamp_sec[i],abs_tol=1.1)) and ( ( (dt.datetime.strptime(timestamp.split("_")[0],"%Y%m%dT%H%M%S")).hour ) == tzero_stamp_hr[i])
        and ( ( (dt.datetime.strptime(timestamp.split("_")[1],"%Y%m%dT%H%M%S")).hour ) == tend_stamp_hr[i])  and ( ( (dt.datetime.strptime(timestamp.split("_")[1],"%Y%m%dT%H%M%S")).minute ) == tend_stamp_min[i]) 
        and (math.isclose( ( (dt.datetime.strptime(timestamp.split("_")[1],"%Y%m%dT%H%M%S")).second ),tend_stamp_sec[i],abs_tol=1.1)) and ( ( (dt.datetime.strptime(timestamp.split("_")[1],"%Y%m%dT%H%M%S")).hour ) == tend_stamp_hr[i]) 
        and ( done == False)  ):
            done = True
            fname = 'MethaneAIR_L1B_O2_'+str(tzero_stamp[i])+'_'+str(tend_stamp[i])+'_'+str(tproc_stamp[i])+'_absolute_correction_akaze.txt'
            data = np.genfromtxt(os.path.join(l1_abs_akaze_DataDir,fname),delimiter=',')
            abs_sc_lon=float(data[0])
            abs_sc_lat=float(data[1])
            abs_off_lon=float(data[2])
            abs_off_lat=float(data[3])
            abs_rlon=float(data[4])
            abs_rlat=float(data[5])
            
    
    if(done==False):
        print('Failed to Find Matching O2 Absolute Shifts')
        exit()
    #########################################################################################
    
    if((abs(rval_lon - 1) >= 0.05 ) or (abs(rval_lat - 1) >= 0.05 ) or  ( abs(abs_rlon - 1) >= 0.05    )    or   ( abs(abs_rlat - 1) >= 0.05      )    ):
        print('Akaze failed for one granule - trying to append granule')
        cfile = 'MethaneAIR_L1B_CH4_'+str(timestamp)+'_relative_correction_akaze.txt'
        os.remove(cfile)
        here = os.getcwd()
        direct1 = str(here).split('ch4_camera_')[0]
        isdir = os.path.isdir(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule')))
        print('isdir+1 = ',isdir)
        if (isdir == True):
            if (os.listdir(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule'))) == []): 
                isdir = os.path.isdir(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule')))    
                if (isdir == True):
                    if (os.listdir(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule'))) == []): 
                        print('No granule before/after to add to akaze - exiting') 
                        exit()
                    else:
                        #find matching o2 granule for iter -1
                        t = str(os.path.basename(str(os.listdir(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule')))[0])).split("MethaneAIR_L1B_CH4_")[1])
                        t = t.split('_')
                        ch4_tzero = t[0]
                        ch4_tend = t[1]
                        ch4_year = ch4_tzero[0:4]
                        ch4_month = ch4_tzero[4:6]
                        ch4_day = ch4_tzero[6:8]
                        ch4_hour = ch4_tzero[9:11]
                        ch4_minute = ch4_tzero[11:13]
                        ch4_second = int(ch4_tzero[13:15])
                        
                        for folder in os.listdir(o2dir):
                            if(folder.endswith('seq')):
                                f = folder.split(".seq")[0]
                                f = f.split("o2_camera_")[1]
                                f = f.split("_")
                                fyear = f[0]
                                fmonth = f[1]
                                fday = f[2]
                                fhour = f[3]
                                fminute = f[4]
                                fsecond = int(f[5])
                                if(( fyear == ch4_year) and (fminute==ch4_minute)and(fhour==ch4_hour)and( math.isclose(fsecond,ch4_second,abs_tol=1.1)) and(fday==ch4_day)and(fmonth==ch4_month)):
                                   filedir = os.path.join(o2dir+folder+'/AvionicsGranule/')
                                   if(os.listdir(filedir)[0].endswith('.nc')):
                                       file = str(os.listdir(filedir)[0])
                                   o2file.append(os.path.join(filedir,file.split('.nc')[0]))
                                   newfiles = os.listdir(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule')))
                                   ch4file.append(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule'),str(newfiles[0]).split('.nc')[0]))
                                   rel_sc_lat,rel_sc_lon,rel_off_lat,rel_off_lon,rval_lat,rval_lon = akaze_nc_ch4_o2_06_24_2021.main(o2file, ch4file)
                                   rel_off_lon = np.float(rel_off_lon) 
                                   rel_sc_lon  = np.float(rel_sc_lon ) 
                                   rel_off_lat = np.float(rel_off_lat) 
                                   rel_sc_lat  = np.float(rel_sc_lat )
                                   if(   (abs(abs_rlon-1) >= 0.05) or (abs(abs_rlat-1) >= 0.05)):
                                       newtimestamp = str(newfiles[0]).split('MethaneAIR_L1B_CH4_')[1] 
                                       t = newtimestamp.split("_")
                                       nfiles_o2 = len(list_abs_files) 
                                       done = False
                                       for i in range(nfiles_o2):
                                           if( ( ( (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S")).hour ) == tzero_stamp_hr[i])  and ( ( (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S")).minute ) == tzero_stamp_min[i]) 
                                           and (math.isclose( ( (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S")).second ),tzero_stamp_sec[i],abs_tol=1.1)) and ( ( (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S")).hour ) == tzero_stamp_hr[i])
                                           and ( ( (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S")).hour ) == tend_stamp_hr[i])  and ( ( (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S")).minute ) == tend_stamp_min[i]) 
                                           and (math.isclose( ( (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S")).second ),tend_stamp_sec[i],abs_tol=1.1)) and ( ( (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S")).hour ) == tend_stamp_hr[i]) 
                                           and ( done == False)  ):
                                               done = True
                                               fname = 'MethaneAIR_L1B_O2_'+str(tzero_stamp[i])+'_'+str(tend_stamp[i])+'_'+str(tproc_stamp[i])+'_absolute_correction_akaze.txt'
                                               data = np.genfromtxt(os.path.join(l1_abs_akaze_DataDir,fname),delimiter=',')
                                               abs_sc_lon=float(data[0])
                                               abs_sc_lat=float(data[1])
                                               abs_off_lon=float(data[2])
                                               abs_off_lat=float(data[3])
                                               abs_rlon=float(data[4])
                                               abs_rlat=float(data[5])
                                       
                                       
                                       if(done==False):
                                           print('Failed to Find Matching O2 Absolute Shifts')
                                           exit()
                                        
                                   
                                   if((abs(rval_lon - 1) >= 0.05 ) or (abs(rval_lat - 1) >= 0.05 ) or  ( abs(abs_rlon - 1) >= 0.05    )    or   ( abs(abs_rlat - 1) >= 0.05      )    ):
                                       print('Akaze failed for multiple granules - exitting')
                                       exit() 
                                   else: 
                                       cfile = 'MethaneAIR_L1B_CH4_'+str(timestamp)+'_relative_correction_akaze.txt'
                                       os.system('cp '+cfile+' '+ l1_rel_akaze_DataDir)
                        
            else:
                #find matching o2 granule for iter +1
                # this is the new ch4 granule
                t = str(os.path.basename(str(os.listdir(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule')))[0])).split("MethaneAIR_L1B_CH4_")[1])
                t = t.split('_')
                ch4_tzero = t[0]
                ch4_tend = t[1]
                ch4_year = ch4_tzero[0:4]
                ch4_month = ch4_tzero[4:6]
                ch4_day = ch4_tzero[6:8]
                ch4_hour = ch4_tzero[9:11]
                ch4_minute = ch4_tzero[11:13]
                ch4_second = int(ch4_tzero[13:15])
                
                for folder in os.listdir(o2dir):
                    if(folder.endswith('seq')):
                        f = folder.split(".seq")[0]
                        f = f.split("o2_camera_")[1]
                        f = f.split("_")
                        fyear = f[0]
                        fmonth = f[1]
                        fday = f[2]
                        fhour = f[3]
                        fminute = f[4]
                        fsecond = int(f[5])
                        if(( fyear == ch4_year) and (fminute==ch4_minute)and(fhour==ch4_hour)and( math.isclose(fsecond,ch4_second,abs_tol=1.1)) and(fday==ch4_day)and(fmonth==ch4_month)):
                           filedir = os.path.join(o2dir+folder+'/AvionicsGranule/')
                           if(os.listdir(filedir)[0].endswith('.nc')):
                               file = str(os.listdir(filedir)[0])
                           o2file.append(os.path.join(filedir,file.split('.nc')[0]))
                           newfiles = os.listdir(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule')))
                           ch4file.append(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule'),str(newfiles[0]).split('.nc')[0]))
                           rel_sc_lat,rel_sc_lon,rel_off_lat,rel_off_lon,rval_lat,rval_lon = akaze_nc_ch4_o2_06_24_2021.main(o2file, ch4file)
                           rel_off_lon = np.float(rel_off_lon) 
                           rel_sc_lon  = np.float(rel_sc_lon ) 
                           rel_off_lat = np.float(rel_off_lat) 
                           rel_sc_lat  = np.float(rel_sc_lat )
                           if(   (abs(abs_rlon-1) >= 0.05) or (abs(abs_rlat-1) >= 0.05)):
                               newtimestamp = str(newfiles[0]).split('MethaneAIR_L1B_CH4_')[1] 
                               t = newtimestamp.split("_")
                               nfiles_o2 = len(list_abs_files) 
                               done = False
                               for i in range(nfiles_o2):
                                   if( ( ( (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S")).hour ) == tzero_stamp_hr[i])  and ( ( (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S")).minute ) == tzero_stamp_min[i]) 
                                   and (math.isclose( ( (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S")).second ),tzero_stamp_sec[i],abs_tol=1.1)) and ( ( (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S")).hour ) == tzero_stamp_hr[i])
                                   and ( ( (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S")).hour ) == tend_stamp_hr[i])  and ( ( (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S")).minute ) == tend_stamp_min[i]) 
                                   and (math.isclose( ( (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S")).second ),tend_stamp_sec[i],abs_tol=1.1)) and ( ( (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S")).hour ) == tend_stamp_hr[i]) 
                                   and ( done == False)  ):
                                       done = True
                                       fname = 'MethaneAIR_L1B_O2_'+str(tzero_stamp[i])+'_'+str(tend_stamp[i])+'_'+str(tproc_stamp[i])+'_absolute_correction_akaze.txt'
                                       data = np.genfromtxt(os.path.join(l1_abs_akaze_DataDir,fname),delimiter=',')
                                       abs_sc_lon=float(data[0])
                                       abs_sc_lat=float(data[1])
                                       abs_off_lon=float(data[2])
                                       abs_off_lat=float(data[3])
                                       abs_rlon=float(data[4])
                                       abs_rlat=float(data[5])
                               
                               
                               if(done==False):
                                   print('Failed to Find Matching O2 Absolute Shifts')
                                   exit()
                           if((abs(rval_lon - 1) >= 0.05 ) or (abs(rval_lat - 1) >= 0.05 ) or  ( abs(abs_rlon - 1) >= 0.05    )    or   ( abs(abs_rlat - 1) >= 0.05      )    ):
                               print('Akaze failed for multiple granules - exitting')
                               exit() 
                           else: 
                               cfile = 'MethaneAIR_L1B_CH4_'+str(timestamp)+'_relative_correction_akaze.txt'
                               os.system('cp '+cfile+' '+ l1_rel_akaze_DataDir)
    
    
    #os.remove(os.path.join(avionics_dir,'MethaneAIR_L1B_CH4_'+timestamp+'.nc'))
    #########################################################################################
    if(done == False):
        print('No absolute correction parameters matched - exit')
        exit()
    #########################################################################################
    #########################################################################################
    nframes = granule.nFrame
    #########################################################################################
    #SPLIT THE GRANULES INTO 10 SECOND INTERVALS 
    granuleList = m.F_cut_granule(granule,granuleSeconds=10)
    #########################################################################################
    #########################################################################################
    
    for i in range(len(granuleList)):
        timestamp = np.min(granuleList[i].frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                                  +np.max(granuleList[i].frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                                  +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
        m.F_save_L1B_mat(timestamp,granuleList[i],headerStr='MethaneAIR_L1B_CH4_')
        filename = os.path.join(cwd,'MethaneAIR_L1B_CH4_'+timestamp+'.nc') 
        #############################################################
    
        if(computer == 'Odyssey'):
            # LOAD OPTIMIZED ORTHORECTIFICATION R CODE
            robjects.r('''source('Orthorectification_Optimized_NC_Fast_CH4.R')''')
            r_getname = robjects.globalenv['Orthorectification_Optimized_NC']
            r_getname(file_flightnc = os.path.join(root_data,flight_nc_file),\
            file_dem = 'dem.tiff',\
            file_l1_o2=filename,\
            dir_output = savepath,\
            framerate               = 0.1,\
            points_x                = 1280,\
            points_sample_optimize  = 3,\
            times_sample            = 3,\
            slope_lon_relative      = rel_sc_lon,\
            slope_lat_relative      = rel_sc_lat,\
            intercept_lon_relative  = rel_off_lon,\
            intercept_lat_relative  = rel_off_lat,\
            slope_lon_absolute      = abs_sc_lon,\
            slope_lat_absolute      = abs_sc_lat,\
            intercept_lon_absolute  = abs_off_lon,\
            intercept_lat_absolute  = abs_off_lat,\
            latvar                  = "LATC",\
            lonvar                  = "LONC",\
            file_eop = os.path.join(root_data,eop_file),\
            dir_lib = os.path.join(root_data,'0User_Functions'),\
            FOV = 33.7)
        else:
            #If computer == Hydra
            args1 =str(rel_sc_lon)
            args2 =str(rel_sc_lat )
            args3 =str(rel_off_lon)
            args4 =str(rel_off_lat )
            args5 =str(abs_sc_lon)
            args6 =str(abs_sc_lat )
            args7 =str(abs_off_lon)
            args8 =str(abs_off_lat )
    
            args9 =str(os.path.join(root_data,flight_nc_file))
            args10=str('dem.tiff')
            args11=str(filename)
            args12=str(savepath)
            args13=str(os.path.join(root_data,eop_file))
            args14=str(os.path.join(root_data,'0User_Functions'))
            subprocess.call(['/home/econway/anaconda3/envs/r4-base/bin/Rscript', 'Orthorectification_Optimized_NC_Fast_CH4_Hydra.R', args1, args2, args3, args4, args5, args6,args7, args8, args9, args10, args11, args12, args13,args14])
    
        x = m.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(l1DataDir),xtrk_aggfac=1,atrk_aggfac=1,ortho_step='avionics')
        os.remove(os.path.join(cwd,'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'))
        name = 'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'
        filename = os.path.join(l1DataDir,name)
        aggregate.main(filename,l1AggDataDir)
    
    
    
    print('Total Time(s) for ',nframes,' Frames = ' ,(time.time() - t_start))

    
elif(molecule == 'o2'):    
    
    ###############################
    #                             #
    #     OXYGEN PROCESSING       #
    #                             #
    ###############################
    
    # test dark and one native granule for CH4 camera
    o = MethaneAIR_L1(whichBand='O2',
    l0DataDir=l0DataDirO2,
    l1DataDir=l1DataDir,
    badPixelMapPath=badPixelMapPathO2,
    radCalPath=radCalPathO2,
    wavCalPath=wavCalPathO2,
    windowTransmissionPath=windowTransmissionPath,
    solarPath=solarPathO2,
    pixellimit=pixellimitO2,
    pixellimitX=pixellimitXO2,
    spectraReffile=spectraReffileO2,
    refitwavelength=refitwavelengthO2,
    calibratewavelength=calibratewavelengthO2,
    ISRFtype=isrftypeO2,
    fitisrf=fitisrfO2,
    calidata = calidatadir,
    fitSZA = fitSZAO2,
    SolZA = SZA,
    ALT = altitude,
    xtol=xtol,
    ftol=ftol,
    xtrackaggfactor=xfac,
    atrackaggfactor=yfac)
    #STRAY LIGHT CORRECTION         
    #o.F_stray_light_input(strayLightKernelPath=strayLightKernelPathO2,rowExtent=200,colExtent=300)
    
    
    dark = o.F_grab_dark(darkfile)
    #########################################################################################
    # SET CURRENT WORKING DIRECTORY AND OUTPUT LOCATION
    #########################################################################################
    cwd = os.getcwd()
    
    if not os.path.exists(os.path.join(cwd,'0Outputs')):
        os.makedirs(os.path.join(cwd,'0Outputs'))
    
    savepath = os.path.join(cwd,'0Outputs')
    
    #########################################################################################
    #PROCESS GRANULE WITH DARK DATA SUBTRACTED
    #########################################################################################
    granule = o.F_granule_processor(os.path.join(l0DataDirO2,targetfile),dark,ePerDN=4.6)
    
    timestamp = np.min(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                              +np.max(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                              +dt.datetime.now().strftime('%Y%m%dT%H%M%S')


    if(flight == 'RF01'):
        flight_nc_file = '0Inputs/MethaneAIRrf01_hrt.nc'
        eop_file = '0Inputs/eop19622020.txt'
    elif(flight == 'RF02'):
        flight_nc_file = '0Inputs/MethaneAIRrf02_hrt.nc'
        eop_file = '0Inputs/eop19622020.txt'

    mintime = np.min(granule.frameDateTime)
    maxtime = np.max(granule.frameDateTime)

    FOV=33.7
    buff = 15000
    dem_file = dem_maker(os.path.join(root_data,flight_nc_file),mintime,maxtime,datenow,FOV,buff) 

    os.system("gdal_translate -of GTiff dem.nc dem.tiff")

    filename = os.path.join(cwd,'MethaneAIR_L1B_O2_'+timestamp+'.nc')
    #########################################################################################
    # SAVE ENTIRE .NC GRANULE
    #########################################################################################
    o.F_save_L1B_mat(timestamp,granule,headerStr='MethaneAIR_L1B_O2_')
    #########################################################################################
    #LOAD GEOLOCATION FUNCTION IN R
    #########################################################################################
    # If computer is Hydra
    if(computer == 'Hydra'):
        args1=str(os.path.join(root_data,flight_nc_file))
        args2=str('dem.tiff')
        args3=str(filename)
        args4=str(savepath)
        args5=str(os.path.join(root_data,'0Inputs/eop19622020.txt'))
        args6=str(os.path.join(root_data,'0User_Functions/'))
        subprocess.call(['/home/econway/anaconda3/envs/r4-base/bin/Rscript', 'Orthorectification_Avionics_NC_Fast_Hydra.R', args1, args2, args3, args4, args5, args6])
    else:
        robjects.r('''source('Orthorectification_Avionics_NC_Fast.R')''')
        r_getname = robjects.globalenv['Orthorectification_Avionics_NC']
        # Avionics Only Routine
        r_getname(file_flightnc = os.path.join(root_data,flight_nc_file),\
        file_dem = 'dem.tiff',\
        file_L1=filename,\
        dir_output = savepath,\
        latvar = "LATC",\
        lonvar = "LONC",\
        file_eop = os.path.join(root_data,eop_file),\
        dir_lib = os.path.join(root_data,'0User_Functions/'),\
        FOV = 33.7)
    #########################################################################################
    # WRITE SPLAT FORMATTED L1B FILE - ENTIRE GRANULE 
    #########################################################################################
    # need to save the avionics only full-granule to a temporary scratch directpry for ch4 cross-alignment
    if not os.path.exists(os.path.join(cwd,'AvionicsGranule')):
        os.makedirs(os.path.join(cwd,'AvionicsGranule'))
    avionics_dir = os.path.join(cwd,'AvionicsGranule')
    
    x = o.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(avionics_dir),xtrk_aggfac=1,atrk_aggfac=1,ortho_step='avionics')
    os.remove(os.path.join(cwd,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))
    os.remove(os.path.join(savepath,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))
    #########################################################################################
    # NEED TO CALL AKAZE ALGORITHM HERE
    files=[]
    files.append(os.path.join(avionics_dir,('MethaneAIR_L1B_O2_'+timestamp))) 
    abs_sc_lat,abs_sc_lon,abs_off_lat,abs_off_lon,rval_lat,rval_lon = akaze_nc_06_24_2021.main(files)
    
    #########################################################################################
    rel_off_lon = np.float(0.0) 
    rel_sc_lon  = np.float(1.0) 
    rel_off_lat = np.float(0.0) 
    rel_sc_lat  = np.float(1.0)
    
    abs_off_lon = np.float(abs_off_lon) 
    abs_sc_lon  = np.float(abs_sc_lon ) 
    abs_off_lat = np.float(abs_off_lat) 
    abs_sc_lat  = np.float(abs_sc_lat )
    
    files=None
    
    if((abs(rval_lon - 1) >= 0.05 ) or (abs(rval_lat - 1) >= 0.05 )   ):
        print('Akaze has failed with one granule - trying to add more frames')
        afile = 'MethaneAIR_L1B_O2_'+str(timestamp)+'_absolute_correction_akaze.txt'
        os.remove(afile)
        files=[]
        file0 = os.path.join(avionics_dir,('MethaneAIR_L1B_O2_'+timestamp))
        files.append(file0)
        here = os.getcwd()
        direct1 = str(here).split('o2_camera_')[0]
        isdir = os.path.isdir(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule')))
        print('isdir1 = ',os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule')),' = ',isdir)
        if (isdir == True):
            print(os.listdir(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule'))))
            if (os.listdir(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule'))) == []): 
                isdir = os.path.isdir(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule')))    
                print('isdir2 = ',os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule')),' = ',isdir)
                if (isdir == True):
                    print(os.listdir(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule'))))
                    if (os.listdir(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule'))) == []): 
                        print('No Granules pre/post current granule for akaze') 
                        exit() 
                    else:
                        newfiles = os.listdir(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule')))
                        files.append(os.path.join(direct1,filenames[int(iteration-1)],str('AvionicsGranule'),str(newfiles[0]).split('.nc')[0]))
                        abs_sc_lat,abs_sc_lon,abs_off_lat,abs_off_lon,rval_lat,rval_lon = akaze_nc_06_24_2021.main(files)
                        afile = 'MethaneAIR_L1B_O2_'+str(timestamp)+'_absolute_correction_akaze.txt'
                        os.system('cp '+afile+' '+ l1_abs_akaze_DataDir)
                        abs_off_lon = np.float(abs_off_lon) 
                        abs_sc_lon  = np.float(abs_sc_lon ) 
                        abs_off_lat = np.float(abs_off_lat) 
                        abs_sc_lat  = np.float(abs_sc_lat )
                        print("Rvalue Lon = ",rval_lon)
                        print("Rvalue Lat = ",rval_lat)
            else:
                newfiles = os.listdir(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule')))
                files.append(os.path.join(direct1,filenames[int(iteration+1)],str('AvionicsGranule'),str(newfiles[0]).split('.nc')[0]))
                abs_sc_lat,abs_sc_lon,abs_off_lat,abs_off_lon,rval_lat,rval_lon = akaze_nc_06_24_2021.main(files)
                afile = 'MethaneAIR_L1B_O2_'+str(timestamp)+'_absolute_correction_akaze.txt'
                os.system('cp '+afile+' '+ l1_abs_akaze_DataDir)
                abs_off_lon = np.float(abs_off_lon) 
                abs_sc_lon  = np.float(abs_sc_lon ) 
                abs_off_lat = np.float(abs_off_lat) 
                abs_sc_lat  = np.float(abs_sc_lat )
                print("Rvalue Lon = ",rval_lon)
                print("Rvalue Lat = ",rval_lat)
                 
    else:
        afile = 'MethaneAIR_L1B_O2_'+str(timestamp)+'_absolute_correction_akaze.txt'
        os.system('cp '+afile+' '+ l1_abs_akaze_DataDir)
        print("Rvalue Lon = ",rval_lon)
        print("Rvalue Lat = ",rval_lat)
    
    
    if((abs(rval_lon - 1) >= 0.05 ) or (abs(rval_lat - 1) >= 0.05 )   ):
        print('Akaze has failed with multiple granules - exitting')
        exit()
    #########################################################################################
    nframes = granule.nFrame
    #########################################################################################
    #SPLIT THE GRANULES INTO 10 SECOND INTERVALS 
    #########################################################################################
    granuleList = o.F_cut_granule(granule,granuleSeconds=10)
    #########################################################################################
    #########################################################################################
    # LOAD OPTIMIZED ORTHORECTIFICATION R CODE
    #########################################################################################
    
    for i in range(len(granuleList)):
        timestamp = np.min(granuleList[i].frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                                  +np.max(granuleList[i].frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                                  +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
        o.F_save_L1B_mat(timestamp,granuleList[i],headerStr='MethaneAIR_L1B_O2_')
        filename = os.path.join(cwd,'MethaneAIR_L1B_O2_'+timestamp+'.nc') 
        #############################################################
        # Avionics Optimized Routine
        #############################################################
    
        if(computer == 'Odyssey'):
            #robjects.r('''source('Orthorectification_Optimized_NC_Fast.R')''')
            robjects.r('''source('Orthorectification_Optimized_NC_Fast_O2.R')''')
            r_getname = robjects.globalenv['Orthorectification_Optimized_NC']
            r_getname(file_flightnc = os.path.join(root_data,flight_nc_file),\
            file_dem = 'dem.tiff',\
            file_l1_o2=filename,\
            dir_output = savepath,\
            framerate               = 0.1,\
            points_x                = 1280,\
            points_sample_optimize  = 3,\
            times_sample            = 3,\
            slope_lon_relative      = rel_sc_lon,\
            slope_lat_relative      = rel_sc_lat,\
            intercept_lon_relative  = rel_off_lon,\
            intercept_lat_relative  = rel_off_lat,\
            slope_lon_absolute      = abs_sc_lon,\
            slope_lat_absolute      = abs_sc_lat,\
            intercept_lon_absolute  = abs_off_lon,\
            intercept_lat_absolute  = abs_off_lat,\
            latvar                  = "LATC",\
            lonvar                  = "LONC",\
            file_eop = os.path.join(root_data,eop_file),\
            dir_lib = os.path.join(root_data,'0User_Functions'),\
            FOV = 33.7)
        else:
            #If computer == Hydra
            args1 =str(rel_sc_lon)
            args2 =str(rel_sc_lat )
            args3 =str(rel_off_lon)
            args4 =str(rel_off_lat )
            args5 =str(abs_sc_lon)
            args6 =str(abs_sc_lat )
            args7 =str(abs_off_lon)
            args8 =str(abs_off_lat )
    
            args9 =str(os.path.join(root_data,flight_nc_file))
            args10=str('dem.tiff')
            args11=str(filename)
            args12=str(savepath)
            args13=str(os.path.join(root_data,eop_file))
            args14=str(os.path.join(root_data,'0User_Functions'))
            subprocess.call(['/home/econway/anaconda3/envs/r4-base/bin/Rscript', 'Orthorectification_Optimized_NC_Fast_O2_Hydra.R', args1, args2, args3, args4, args5, args6,args7, args8, args9, args10, args11, args12, args13,args14])
    
        x = o.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(l1DataDir),xtrk_aggfac=1,atrk_aggfac=1,ortho_step='avionics')
        os.remove(os.path.join(cwd,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))
        name = 'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'
        filename = os.path.join(l1DataDir,name)
        aggregate.main(filename,l1AggDataDir)
    
    print('Total Time(s) for ',nframes,' Frames = ' ,(time.time() - t_start))
