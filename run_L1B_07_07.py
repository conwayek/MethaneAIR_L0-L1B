"""
Created on Thu Sep 29 14:25:00 2020# -*- coding: utf-8 -*-
@authors: kangsun & eamonconway
"""
from filelock import FileLock
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
import aggregate
import dem_maker 
from GEOAkaze import GEOAkaze

def main():
    # Update the following for your local environment
    flight = 'RF01'
    computer = 'Odyssey' # Hydra or Odyssey
    molecule = 'ch4' # ch4 or o2
    submission = 'array' # array or serial
    priority = int(1) # priority index of data = 0 (not priority) or 1 (priority)
    file_landsat = []
    file_landsat.append('/n/holylfs04/LABS/wofsy_lab/Lab/econway/MSI_B11')
    root_dest = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/'
    root_data = '/n/holylfs04/LABS/wofsy_lab/Lab/econway/DATA/'
    hydra_base_dir = '/scratch/sao_atmos/econway/'
    odyssey_base_dir1 = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/data/flight_testing/' 
    odyssey_base_dir2 = '/n/wofsy_lab/MethaneAIR/flight_data/' 
    odyssey_base_dir3 = '/n/wofsyfs2/MethaneAIR/flight_data/' 
    avionics_only = False
    MSI_Climatology =  '/n/holylfs04/LABS/wofsy_lab/Lab/econway/MSI_Clim'
    
    # This is the date of the flight
    
    if(flight == 'RF01'):
        datenow = dt.datetime(year = 2019, month = 11, day = 8)
    elif(flight == 'RF02'):
        datenow = dt.datetime(year = 2019, month = 11, day = 12)
    elif(flight == 'CheckOut'):
        datenow = dt.datetime(year = 2021, month = 7, day = 28)
    elif(flight == 'RF04'):
        datenow = dt.datetime(year = 2021, month = 7, day = 30)
    elif(flight == 'RF05'):
        datenow = dt.datetime(year = 2021, month = 8, day = 3)
    elif(flight == 'RF06'):
        datenow = dt.datetime(year = 2021, month = 8, day = 6)
    elif(flight == 'RF07'):
        datenow = dt.datetime(year = 2021, month = 8, day = 9)
    elif(flight == 'RF08'):
        datenow = dt.datetime(year = 2021, month = 8, day = 11)
    elif(flight == 'RF09'):
        datenow = dt.datetime(year = 2021, month = 8, day = 13)
    elif(flight == 'RF10'):
        datenow = dt.datetime(year = 2021, month = 8, day = 16)
    
    
    if(computer == 'Odyssey'):
    
        if(flight == 'RF01'):
            ac_file = 'MethaneAIRrf01_hrt.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir1,'20191108/ch4_camera/')
                seq_files = os.path.join(root_data,'files.ch4.rf01.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir1,'20191108/o2_camera/')
                seq_files = os.path.join(root_data,'files.o2.rf01.txt')
    
    
        elif(flight == 'RF02'):
            ac_file = 'MethaneAIRrf02_hrt.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir1,'20191112/ch4_camera/')
                seq_files = os.path.join(root_data,'ch4_seq_files_RF02.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir1,'20191112/o2_camera/')
                seq_files = os.path.join(root_data,'o2_seq_files_RF02.txt')
    
    
        elif(flight == 'CheckOut'):
            ac_file = 'MethaneAIR21rf03h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir2,'20210728/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_ch4_seq_files_RF03.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_ch4_seq_files_RF03.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir2,'20210728/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_o2_seq_files_RF03.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_o2_seq_files_RF03.txt')
    
        elif(flight == 'RF04'):
            ac_file = 'MethaneAIR21rf04h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir2,'20210730/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_ch4_seq_files_RF04.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_o2_seq_files_RF04.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir2,'20210730/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_o2_seq_files_RF04.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_o2_seq_files_RF04.txt')
    
        elif(flight == 'RF05'):
            ac_file = 'MethaneAIR21rf05h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir2,'20210803/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_ch4_seq_files_RF05.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_ch4_seq_files_RF05.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir2,'20210803/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_o2_seq_files_RF05.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_o2_seq_files_RF05.txt')
    
        elif(flight == 'RF06'):
            ac_file = 'MethaneAIR21rf06h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210806/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_ch4_seq_files_RF06.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_ch4_seq_files_RF06.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210806/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_o2_seq_files_RF06.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_o2_seq_files_RF06.txt')
    
        elif(flight == 'RF07'):
            ac_file = 'MethaneAIR21rf07h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210809/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_ch4_seq_files_RF07.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_ch4_seq_files_RF07.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210809/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_o2_seq_files_RF07.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_o2_seq_files_RF07.txt')
    
    
        elif(flight == 'RF08'):
            ac_file = 'MethaneAIR21rf08h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210811/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_ch4_seq_files_RF08.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_ch4_seq_files_RF08.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210811/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_o2_seq_files_RF08.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_o2_seq_files_RF08.txt')
    
    
        elif(flight == 'RF09'):
            ac_file = 'MethaneAIR21rf09h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210813/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_ch4_seq_files_RF09.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_ch4_seq_files_RF09.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210813/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data,'priority_o2_seq_files_RF09.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data,'non_priority_o2_seq_files_RF09.txt')
    
    
    
        elif(flight == 'RF10'):
            ac_file = 'MethaneAIR21rf10h_reprocessed.nc' 
            if(molecule == 'o2'):
                root_l0 = os.path.join(odyssey_base_dir3,'20210816/o2_camera/')
                seq_files = os.path.join(root_data,'priority_o2_seq_files_RF10.txt')
    
    
        else:
            print('Bad Research Flight')
            exit()
    
    
    elif(computer == 'Hydra'):
    
        if(flight == 'RF01'):
            ac_file = 'MethaneAIRrf01_hrt.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(hydra_base_dir, 'MethaneAIR_L0_RF01_CH4/')
                seq_files = os.path.join(root_data,'files.ch4.rf01.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(hydra_base_dir, 'MethaneAIR_L0_RF01_O2/')
                seq_files = os.path.join(root_data,'files.o2.rf01.txt')
    
        elif(flight == 'RF06'):
            ac_file = 'MethaneAIR21rf06h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(hydra_base_dir, 'RF06/L0_DATA_RF06/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_ch4_seq_files_RF06.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_ch4_seq_files_RF06.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(hydra_base_dir, 'RF06/L0_DATA_RF06/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_o2_seq_files_RF06.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_o2_seq_files_RF06.txt')
    
        elif(flight == 'RF07'):
            ac_file = 'MethaneAIR21rf07h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(hydra_base_dir, 'RF07/L0_DATA_RF07/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_ch4_seq_files_RF07.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_ch4_seq_files_RF07.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(hydra_base_dir, 'RF07/L0_DATA_RF07/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_o2_seq_files_RF07.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_o2_seq_files_RF07.txt')
    
        elif(flight == 'RF08'):
            ac_file = 'MethaneAIR21rf08h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(hydra_base_dir, 'RF08/L0_DATA_RF08/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_ch4_seq_files_RF08.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_ch4_seq_files_RF08.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(hydra_base_dir, 'RF08/L0_DATA_RF08/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_o2_seq_files_RF08.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_o2_seq_files_RF08.txt')
    
        elif(flight == 'RF09'):
            ac_file = 'MethaneAIR21rf09h_reprocessed.nc' 
            if(molecule == 'ch4'):
                root_l0 = os.path.join(hydra_base_dir, 'RF09/L0_DATA_RF09/ch4_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_ch4_seq_files_RF09.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_ch4_seq_files_RF09.txt')
            elif(molecule == 'o2'):
                root_l0 = os.path.join(hydra_base_dir, 'RF09/L0_DATA_RF09/o2_camera/')
                if(priority == 1):
                    seq_files = os.path.join(root_data, 'priority_o2_seq_files_RF09.txt')
                elif(priority == 0):
                    seq_files = os.path.join(root_data, 'non_priority_o2_seq_files_RF09.txt')
    
        else:
            print('Bad Research Flight - Not Implemented for - ',flight, str(computer))
            exit()
    
    ## THIS IS NUMBER THE FILE WE ARE GOING TO PROCESS
    if(submission == 'serial'):
        for line in open('iteration.txt', 'r'):
          values = [float(s) for s in line.split()]
          iteration = np.int(values[0])
    elif(submission == 'array'):
        pwd = os.getcwd()
        iteration =int( pwd.split('/')[-1])
        iteration = int(iteration - 1)

    runL1B(flight, computer, molecule, iteration, priority, file_landsat, root_dest, root_data, datenow, ac_file, root_l0, seq_files,MSI_Climatology)    
   

def runL1B(flight, computer, molecule, iteration, priority, file_landsat, root_dest, root_data, datenow, ac_file, root_l0, seq_files,MSI_Climatology):

    t_start = time.time()
    if(molecule=='o2'):
        l1_GEOakaze_nc_DataDir = os.path.join(root_dest,str(flight)+'_V2/GEOAkazeNC_O2/')
        l1_GEOakaze_kmz_DataDir = os.path.join(root_dest,str(flight)+'_V2/GEOAkazeKMZ_O2/')
        l1_GEOakaze_nc_Fail_DataDir = os.path.join(root_dest,str(flight)+'_V2/GEOAkazeNC_Failed_O2/')
    else:
        l1_GEOakaze_nc_DataDir = os.path.join(root_dest,str(flight)+'_V2/GEOAkazeNC_CH4/')
        l1_GEOakaze_kmz_DataDir = os.path.join(root_dest,str(flight)+'_V2/GEOAkazeKMZ_CH4/')
        l1_GEOakaze_nc_Fail_DataDir = os.path.join(root_dest,str(flight)+'_V2/GEOAkazeNC_Failed_CH4/')
    
    l1_abs_akaze_DataDir = os.path.join(root_dest,str(flight)+'_V2/O2_ABS_AKAZE_FILES/')
    l1_rel_akaze_DataDir = os.path.join(root_dest,str(flight)+'_V2/CH4_REL_AKAZE_FILES/')
    
    o2_native_for_akaze = os.path.join(root_dest,str(flight)+'_V2/O2_NATIVE/')
    
    o2dir = os.path.join(root_dest,str(flight)+'_V2/O2Avionics/')
    o2diragg = os.path.join(root_dest,str(flight)+'_V2/O2Avionics_15x3/')
    o2diragg2 = os.path.join(root_dest,str(flight)+'_V2/O2Avionics_5x1/')
    ch4dir = os.path.join(root_dest,str(flight)+'_V2/CH4Avionics/')
    ch4diragg = os.path.join(root_dest,str(flight)+'_V2/CH4Avionics_15x3/')
    ch4diragg2 = os.path.join(root_dest,str(flight)+'_V2/CH4Avionics_5x1/')
    
    
    if(molecule == 'ch4'):
        l1DataDir = os.path.join(root_dest,str(flight)+'_V2/CH4_NATIVE/')
        l1AggDataDir = os.path.join(root_dest,str(flight)+'_V2/CH4_15x3/')
        l1AggDataDir2 = os.path.join(root_dest,str(flight)+'_V2/CH4_5x1/')
        logfile_native = os.path.join(l1DataDir,'log_file.txt')
        logfile_agg = os.path.join(l1AggDataDir,'log_file.txt')
        logfile_agg2 = os.path.join(l1AggDataDir2,'log_file.txt')
        logfile_avionics_native = os.path.join(ch4dir,'log_file.txt')
        logfile_avionics_agg = os.path.join(ch4diragg,'log_file.txt')
        logfile_avionics_agg2 = os.path.join(ch4diragg2,'log_file.txt')
    if(molecule == 'o2'):
        l1DataDir = os.path.join(root_dest,str(flight)+'_V2/O2_NATIVE/')
        l1AggDataDir = os.path.join(root_dest,str(flight)+'_V2/O2_15x3/')
        l1AggDataDir2 = os.path.join(root_dest,str(flight)+'_V2/O2_5x1/')
        logfile_native = os.path.join(l1DataDir,'log_file.txt')
        logfile_agg = os.path.join(l1AggDataDir,'log_file.txt')
        logfile_agg2 = os.path.join(l1AggDataDir2,'log_file.txt')
        logfile_avionics_native = os.path.join(o2dir,'log_file.txt')
        logfile_avionics_agg = os.path.join(o2diragg,'log_file.txt')
        logfile_avionics_agg2 = os.path.join(o2diragg2,'log_file.txt')
    
    
    
    flight_nc_file = '0Inputs/'+str(ac_file)
    eop_file = '0Inputs/EOP-Last5Years_celestrak_20210723.txt'
    
    
    
    
    windowTransmissionPath = os.path.join(root_data,'window_transmission.mat')
    # CH4 CHANNEL DATA    
    badPixelMapPathCH4 = os.path.join(root_data,'CH4_bad_pix.csv')
    if(flight=='RF01' or flight=='RF02'):
        wavCalPathCH4 = os.path.join(root_data,'methaneair_ch4_spectroscopic_calibration_1280_footprints.nc')
        radCalPathCH4 = os.path.join(root_data,'rad_coef_ch4_100ms_ord4_20200209T211211.mat')
    else:
        wavCalPathCH4 = os.path.join(root_data,'methaneair_ch4_isrf_1280_footprints_20210722T153105.nc')
        radCalPathCH4 = os.path.join(root_data,'rad_coef_ch4_100ms_ord4_20200219T022658.mat')
    l0DataDirCH4 = root_l0
    strayLightKernelPathCH4 = os.path.join(root_data,'K_stable_CH4.mat')
    solarPathCH4= os.path.join(root_data,'hybrid_reference_spectrum_p001nm_resolution_with_unc.nc')
    pixellimitCH4=[200,620]
    pixellimitXCH4=[100,1000]
    #pixellimitXCH4=[100,1000]
    spectraReffileCH4=os.path.join(root_data,'spectra_CH4_channel.nc')
    refitwavelengthCH4 = True
    calibratewavelengthCH4 = True
    fitisrfCH4 = True
    isrftypeCH4 = 'GAUSS' 
    fitSZACH4 = True
    
    # O2 CHANNEL DATA    
    badPixelMapPathO2 = os.path.join(root_data,'O2_bad_pix.csv')
    if(flight=='RF01' or flight=='RF02'):
        wavCalPathO2 = os.path.join(root_data,'methaneair_o2_spectroscopic_calibration_1280_footprints.nc')
        radCalPathO2 = os.path.join(root_data,'rad_coef_o2_100ms_ord4_20200205T230528.mat')
    else:
        wavCalPathO2 = os.path.join(root_data,'methaneair_o2_isrf_1280_footprints_20210722T153733.nc')
        radCalPathO2 = os.path.join(root_data,'rad_coef_o2_100ms_ord4_20200310T033130.mat')
    l0DataDirO2 = root_l0
    strayLightKernelPathO2 = os.path.join(root_data,'K_stable_O2.mat')
    solarPathO2= os.path.join(root_data,'hybrid_reference_spectrum_p001nm_resolution_with_unc.nc')
    pixellimitO2=[500,570]
    pixellimitXO2=[100,1000]
    spectraReffileO2=os.path.join(root_data,'spectra_O2_channel.nc')
    refitwavelengthO2 = False
    calibratewavelengthO2 = False
    fitisrfO2 = False
    isrftypeO2 = 'ISRF'
    fitSZAO2 = False
    if(flight!='RF10'):
        refitwavelengthO2 = True
        calibratewavelengthO2 = True
        fitisrfO2 = True
        isrftypeO2 = 'GAUSS'
        fitSZAO2 = True
    
    calidatadir = root_data 
    
    xtol = 1e-2
    ftol = 1e-2
    
    fitxfac=30
    
    xfac =15
    yfac = 3 
    
    
    
    
    
    #READ IN ALL THE FILE NAMES FROM THE OBSERVATION
    with open(os.path.join(root_data,seq_files), 'r') as file:
    	filenames = file.read().splitlines()
    file.close()

    targetfile = filenames[iteration]
    
    #####
    # HERE WE GET THE TIME FROM THE GRANULES AND GET THEIR ALTITUDE: ACCURATE TO 100-200 METER OR SO ON AS(DE)CENDING LEGS OF JOURNEY
    #####
    ncid = Dataset(os.path.join(root_data,ac_file), 'r')
    altitude1 = ncid.variables['GGALT'][:]
    altitude2 = ncid.variables['GGEOIDHT'][:]
    timeGPS = ncid.variables['Time'][:]
    solze = ncid.variables['SOLZE'][:,0]
    
    height = np.zeros(len(timeGPS))
    height = altitude1 + altitude2
    
    
    temp = targetfile.split('_camera_')[1]
    temp = dt.datetime.strptime(temp.strip('.seq'),'%Y_%m_%d_%H_%M_%S')
    seconds = (temp - datenow).total_seconds()
    
    
    SZA_interp = interpolate.interp1d(timeGPS,solze) 
    ALT_interp = interpolate.interp1d(timeGPS,height)
    
    SZA = SZA_interp(seconds)
    altitude =  ALT_interp(seconds)
    
    if(SZA < 0 or altitude < 0 ):
        # Set default values
        SZA = 60.0*3.14159/180.0 
        altitude = 10000
        
    
    
    if(flight == 'RF01'):
        if(altitude <= (25000*0.3048)  ):
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
    
    else:
        import darkfinder
        darkfiles = darkfinder.main(root_l0,str(flight))
       
        darkfiles = np.array(darkfiles)
        nfiles = len(darkfiles)
        times = np.zeros(len(darkfiles))
        
        for i in range(nfiles):
            t = (os.path.basename(darkfiles[i])).split('_camera_')[1]
            t = dt.datetime.strptime(t.strip('.seq'),'%Y_%m_%d_%H_%M_%S')
            times[i] = (t - dt.datetime(1985,1,1,0,0,0)).total_seconds()
    
        idx = np.argsort(times)
        darkfiles = darkfiles[idx]
        # get the current L0 filetime 
        temp = targetfile.split('_camera_')[1]
        time_now = dt.datetime.strptime(temp.strip('.seq'),'%Y_%m_%d_%H_%M_%S')    
        darkfile_times = {}
        num_darkfiles = len(darkfiles)
        valid_times = []
        for i in range(num_darkfiles):
            t = (os.path.basename(darkfiles[i])).split('_camera_')[1]
            t = dt.datetime.strptime(t.strip('.seq'),'%Y_%m_%d_%H_%M_%S')
            # If current darkfile time is before L0 filetime
            # store it to check later
            if t <= time_now:
                valid_times.append(t)
                # map parsed file time to darkfile name
                darkfile_times[t] = darkfiles[i]
        
        if len(valid_times) < 1:
            darkfile = darkfiles[0]      
        else: 
            # After sorting the last element in the array 
            # is the latest darkfile before current L0 file time
            valid_times.sort()
            darkfile = darkfile_times[valid_times[-1]]   
    
    #########################################################################################
    #########################################################################################
    
    
    from MethaneAIR_L1_EKC_07_08_2021 import MethaneAIR_L1
    
    if(molecule == 'ch4'):
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
        xtrackaggfactor=fitxfac)
                          
        
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
    
        mintime = np.min(granule.frameDateTime)
        maxtime = np.max(granule.frameDateTime)
    
        # new dem maker
        FOV=33.7
        buff = 15000
        dem_file = dem_maker.main(os.path.join(root_data,flight_nc_file),mintime,maxtime,datenow,FOV,buff) 
    
        filename = os.path.join(cwd,'MethaneAIR_L1B_CH4_'+timestamp+'.nc')
        #########################################################################################
        # SAVE ENTIRE .NC GRANULE
        m.F_save_L1B_mat(timestamp,granule,headerStr='MethaneAIR_L1B_CH4_')
        #########################################################################################
        #LOAD GEOLOCATION FUNCTION IN R
        # If computer is Hydra
        if(computer == 'Hydra'):
            args1=str(os.path.join(root_data,flight_nc_file))
            args2=str(dem_file)
            args3=str(filename)
            args4=str(savepath)
            args5=str(os.path.join(root_data,'0Inputs/EOP-Last5Years_celestrak_20210723.txt'))
            args6=str(os.path.join(root_data,'0User_Functions/'))
            args7=str("LATC")
            args8=str("LONC")
            args9=str("PITCH")
            args10=str("ROLL")
            args11=str("THDG")
            args12=str("GEOPTH")
            args13=str("GGEOIDHT")
    
    
    
            subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__), 'Orthorectification_Avionics_NC_Fast_Hydra.R'), args1, args2, args3, args4, args5, args6,args7,args8,args9,args10,args11,args12,args13])
        else:
            args1=str(os.path.join(root_data,flight_nc_file))
            args2=str(dem_file)
            args3=str(filename)
            args4=str(savepath)
            args5=str(os.path.join(root_data,'0Inputs/EOP-Last5Years_celestrak_20210723.txt'))
            args6=str(os.path.join(root_data,'0User_Functions/'))
            args7=str("LATC")
            args8=str("LONC")
            args9=str("PITCH")
            args10=str("ROLL")
            args11=str("THDG")
            args12=str("GEOPTH")
            args13=str("GGEOIDHT")
    
            subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Avionics_NC_Fast_Hydra.R'), args1, args2, args3, args4, args5, args6,args7,args8,args9,args10,args11,args12,args13])
        #########################################################################################
        # WRITE SPLAT FORMATTED L1B FILE - ENTIRE GRANULE 
        # need to save the avionics only full-granule to a temporary scratch directpry for ch4 cross-alignment
        if not os.path.exists(os.path.join(cwd,'AvionicsGranule')):
            os.makedirs(os.path.join(cwd,'AvionicsGranule'))
        avionics_dir = os.path.join(cwd,'AvionicsGranule')
    
        x = m.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(avionics_dir),1,1,'avionics',nframe0=None,nframe1=None,avfile=None)
        os.remove(os.path.join(cwd,'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'))
        os.remove(os.path.join(savepath,'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'))
        os.system('cp '+str(os.path.join(avionics_dir,'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'))+' '+ str(ch4dir)) 
        akaze_ch4_file = os.path.join(ch4dir,'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc') 
        """
        name = 'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'
        filename = os.path.join(avionics_dir,name)
        aggregate.main(filename,ch4diragg,xfac,yfac,av=True)
        
        name = 'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'
        filename = os.path.join(avionics_dir,name)
        aggregate.main(filename,ch4diragg2,5,1,av=True)
    
        lockname=logfile_avionics_native+'.lock'
        with FileLock(lockname):
            f = open(logfile_avionics_native,'a+') 
            f.write(str(os.path.join(ch4dir,'MethaneAIR_L1B_CH4_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
            f.close()
        lockname=logfile_avionics_agg+'.lock'
        with FileLock(lockname):
            f = open(logfile_avionics_agg,'a+') 
            f.write(str(os.path.join(ch4diragg,'MethaneAIR_L1B_CH4_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
            f.close()
        lockname=logfile_avionics_agg2+'.lock'
        with FileLock(lockname):
            f = open(logfile_avionics_agg2,'a+') 
            f.write(str(os.path.join(ch4diragg2,'MethaneAIR_L1B_CH4_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
            f.close()
        """
    
        #########################################################################################
        # NEED TO CALL AKAZE ALGORITHM HERE - NEED RELATIVE ADJUSTMENT PARAMETERS
        #########################################################################################
        # Current CH4 granule times
        t = timestamp.split("_")
        ch4_tzero = t[0]
        ch4_tend = t[1]
        ch4_year = int(ch4_tzero[0:4])
        ch4_month = int(ch4_tzero[4:6])
        ch4_day = int(ch4_tzero[6:8])
        ch4_hour = int(ch4_tzero[9:11])
        ch4_minute = int(ch4_tzero[11:13])
        ch4_second = int(ch4_tzero[13:15])
        ch4_hour_start = (dt.datetime.strptime(t[0],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
        ch4_hour_end = (dt.datetime.strptime(t[1],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
        
        o2file=[]
        for folder in os.listdir(o2_native_for_akaze):
            if(folder.endswith('nc')):
                f = folder.split(".nc")[0]
                f = f.split("MethaneAIR_L1B_O2_")[1]
                fstart = (dt.datetime.strptime(f.split("_")[0],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                fend = (dt.datetime.strptime(f.split("_")[1],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
        
                if( ( math.isclose(fstart,ch4_hour_start,abs_tol=1.1 ) )):
                   o2file.append(os.path.join(o2_native_for_akaze,folder))
                elif( ( math.isclose(fend,ch4_hour_end,abs_tol=1.1 ) )):
                   o2file.append(os.path.join(o2_native_for_akaze,folder))
                elif( (fend<ch4_hour_end ) and (fstart>ch4_hour_start)  ):
                   o2file.append(os.path.join(o2_native_for_akaze,folder))
        
        ch4file = []
        ch4file.append(akaze_ch4_file)
        if(o2file == []):
            print('No Matched O2 File')
            avionics_only = True 
        # Geoakaze Implementation:
        # slave = ch4 (avionics)
        # master= o2 (avionics)
        if(avionics_only == False):
            try:
                gkobj = GEOAkaze(ch4file, o2file, 0.0001, 0, 0, 10000,msi_clim_fld = MSI_Climatology, is_histeq=True, is_destriping=False, bandindex_slave=1,bandindex_master=1, w1=0, w2=160,w3=100,w4=450)
                gkobj.readslave()
                gkobj.readmaster()
                gkobj.append_master()
                gkobj.akaze()
    
                rel_sc_lat=gkobj.slope_lat
                rel_sc_lon=gkobj.slope_lon
                rel_off_lat=gkobj.intercept_lat
                rel_off_lon=gkobj.intercept_lon
                rval_lat =gkobj.success
                rval_lon = gkobj.success
                if(gkobj.success==1):
                    gkobj.savetokmz(os.path.join(os.getcwd(),'MethaneAIR_L1B_CH4_'+timestamp+'.kmz'))
                    gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_DataDir,'MethaneAIR_L1B_CH4_'+timestamp+'.nc'))
                    gkobj.savetotxt(os.path.join(os.getcwd(),'MethaneAIR_L1B_CH4_'+timestamp))
                    os.system('cp *.kmz* '+ l1_GEOakaze_kmz_DataDir)
                else:
                    try:
                        print('Trying to copy failed GEOAkze nc4 file to failed directory')
                        gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_Fail_DataDir,'MethaneAIR_L1B_CH4_'+timestamp+'.nc'))
                    except:
                        print('Failed to copy GEOAkze nc4 file to failed directory - continuing')
            except:
                try:
                    print('Trying to copy failed GEOAkze nc4 file to failed directory')
                    gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_Fail_DataDir,'MethaneAIR_L1B_CH4_'+timestamp+'.nc'))
                except:
                    print('Failed to copy GEOAkze nc4 file to failed directory - continuing')
                rval_lat = 0
                rval_lon = 0
                avionics_only = True
            
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
                    t=file.split("_correction_factors_akaze.txt")[0]
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
            newtimestart = (dt.datetime.strptime(timestamp.split("_")[0],"%Y%m%dT%H%M%S") - dt.datetime(1985,1,1,0,0,0)).total_seconds()
            newtimeend = (dt.datetime.strptime(timestamp.split("_")[1],"%Y%m%dT%H%M%S") - dt.datetime(1985,1,1,0,0,0)).total_seconds()
            for i in range(nfiles_o2):
                secondsend = tend_stamp_hr[i]*3600.0 + tend_stamp_min[i]*60 + tend_stamp_sec[i]
                secondsstart = tzero_stamp_hr[i]*3600.0 + tzero_stamp_min[i]*60 + tzero_stamp_sec[i]
                tend = (datenow + dt.timedelta(seconds=secondsend) - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                tstart = (datenow + dt.timedelta(seconds=secondsstart) - dt.datetime(1985,1,1,0,0,0)).total_seconds()   
    
                if( (done==False ) and (math.isclose(tend,newtimeend,abs_tol=1.1)) and math.isclose(tstart,newtimestart,abs_tol=1.1)  ):
                    done = True
                    fname = 'MethaneAIR_L1B_O2_'+str(tzero_stamp[i])+'_'+str(tend_stamp[i])+'_'+str(tproc_stamp[i])+'_correction_factors_akaze.txt'
                    data = np.genfromtxt(os.path.join(l1_abs_akaze_DataDir,fname),delimiter=',')
                    abs_sc_lon=float(data[0])
                    abs_sc_lat=float(data[1])
                    abs_off_lon=float(data[2])
                    abs_off_lat=float(data[3])
                    abs_rlon=float(data[4])
                    abs_rlat=float(data[5])
                elif( (done==False ) and ( newtimeend<=tend  ) and (tstart<=newtimestart) ):
                    done = True
                    fname = 'MethaneAIR_L1B_O2_'+str(tzero_stamp[i])+'_'+str(tend_stamp[i])+'_'+str(tproc_stamp[i])+'_correction_factors_akaze.txt'
                    data = np.genfromtxt(os.path.join(l1_abs_akaze_DataDir,fname),delimiter=',')
                    abs_sc_lon=float(data[0])
                    abs_sc_lat=float(data[1])
                    abs_off_lon=float(data[2])
                    abs_off_lat=float(data[3])
                    abs_rlon=float(data[4])
                    abs_rlat=float(data[5])
                    
            if(done==False):
                print('Failed to Find Matching O2 Absolute Shifts')
                avionics_only = True
            try:
                print('O2 AKAZE correction file = ',fname)
            except:
                pass
            #########################################################################################
            
            if((abs(rval_lon - 1) >= 0.05 ) or (abs(rval_lat - 1) >= 0.05 ) or avionics_only == True ):
                print('Akaze failed for one granule - trying to append granule')
                 
                #date_end_before = start of current    
                # Searching for granule before this one
                tstart_now = np.min(granule.frameDateTime)
                tend_now = np.max(granule.frameDateTime)
                tstart_now_seconds = (tstart_now - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                tend_now_seconds = (tend_now - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                matched = False 
                for file in os.listdir(ch4dir):
                    if(file.endswith('.nc')):
                        tnew = file.split('MethaneAIR_L1B_CH4_')[1]
                        tnew = tnew.split('.nc')[0]  
                        tend = tnew.split('_')[1]
                        tend = dt.datetime.strptime(tend,'%Y%m%dT%H%M%S')#.strftime('%Y%m%dT%H%M%S')
                        tend = (tend - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                        tstart = tnew.split('_')[0]
                        tstart = dt.datetime.strptime(tstart,'%Y%m%dT%H%M%S')#.strftime('%Y%m%dT%H%M%S')
                        tstart = (tstart - dt.datetime(1985,1,1,0,0,0)).total_seconds()

                        if( (matched == False) and (math.isclose(tend,tstart_now_seconds,abs_tol=2) ) and (math.isclose(tstart,tstart_now_seconds,abs_tol=1)!=True)):
                            ch4file.append(os.path.join(ch4dir,file)) 
                            matchedch4file = file
                            matched = True
                            t = file.split('MethaneAIR_L1B_CH4_')[1]
                            t = t.split('.nc')[0]
                            appended_ch4_file_start = (dt.datetime.strptime(t[0],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                            appended_ch4_file_end   = (dt.datetime.strptime(t[1],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                        elif( (matched == False) and (math.isclose(tstart,tend_now_seconds,abs_tol=2) ) and (math.isclose(tend,tend_now_seconds,abs_tol=1)!=True)):
                            ch4file.append(os.path.join(ch4dir,file)) 
                            matched = True
                            matchedch4file = file
                            t = file.split('MethaneAIR_L1B_CH4_')[1]
                            t = t.split('.nc')[0]
                            appended_ch4_file_start = (dt.datetime.strptime(t[0],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                            appended_ch4_file_end   = (dt.datetime.strptime(t[1],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
    
                # Search for O2 Avionics File
                if (matched == True):
                    print('Found a second ch4 granule for akaze, now getting a second o2 granule')
                    temp = matchedch4file.split('MethaneAIR_L1B_CH4_')[1]
                    temp = temp.split('.nc')[0] 
                    temp = temp.split('_')
                    tstart_now_seconds = (dt.datetime.strptime(temp[0],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                    tend_now_seconds = (dt.datetime.strptime(temp[1],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                    matchedo2 = False 
                    for file in os.listdir(o2_native_for_akaze):
                        if('.nc' in file):
                            tnew = file.split('MethaneAIR_L1B_O2_')[1]
                            tnew = tnew.split('.nc')[0]  
                            to2 = tnew.split('_')
                            tstart = (dt.datetime.strptime(to2[0],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                            tend = (dt.datetime.strptime(to2[1],'%Y%m%dT%H%M%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                             
                            if( ( math.isclose(tstart,tstart_now_seconds,abs_tol=1.1 ) )):
                                o2file.append(os.path.join(o2_native_for_akaze,file))
                                matchedo2file = file
                                matchedo2 = True
                            elif( ( math.isclose(tend,tend_now_seconds,abs_tol=1.1 ) )):
                                o2file.append(os.path.join(o2_native_for_akaze,file))
                                matchedo2file = file
                                matchedo2 = True
                            elif( (tend<tend_now_seconds ) and (tstart>tstart_now_seconds)  ):
                                o2file.append(os.path.join(o2_native_for_akaze,file)) 
                                matchedo2file = file
                                matchedo2 = True
                else:
                    print('Could not find CH4 granule to append to current granule')
                    rval_lat = 0
                    rval_lon = 0
                    avionics_only = True
                    matchedo2 = False
                if(matchedo2 == False):
                    print('Could not find a second o2 granule')
                    rval_lat = 0
                    rval_lon = 0
                    avionics_only = True
    
    
    
    
                if((matchedo2 == True) and ((matched == True))):
                    try:
                        gkobj = GEOAkaze(ch4file, o2file, 0.0001, 0, 0, 10000,msi_clim_fld = MSI_Climatology, is_histeq=True, is_destriping=False, bandindex_slave=1,bandindex_master=1, w1=0, w2=160,w3=100,w4=450)
                        gkobj.readslave()
                        gkobj.readmaster()
                        gkobj.append_master()
                        gkobj.akaze()
    
    
                        rel_sc_lat=gkobj.slope_lat
                        rel_sc_lon=gkobj.slope_lon
                        rel_off_lat=gkobj.intercept_lat
                        rel_off_lon=gkobj.intercept_lon
                        rval_lat =gkobj.success
                        rval_lon = gkobj.success
                        if(gkobj.success==1):
                            gkobj.savetokmz(os.path.join(os.getcwd(),'MethaneAIR_L1B_CH4_'+timestamp+'.kmz'))
                            gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_DataDir,'MethaneAIR_L1B_CH4_'+timestamp+'.nc'))
                            gkobj.savetotxt(os.path.join(os.getcwd(),'MethaneAIR_L1B_CH4_'+timestamp))
                            os.system('cp *.kmz* '+ l1_GEOakaze_kmz_DataDir)
                            newtimestamp = matchedo2file.split('MethaneAIR_L1B_O2_')[1]
                            newtimestamp = newtimestamp.split('.nc')[0]
                            newtimestart = (dt.datetime.strptime(newtimestamp.split("_")[0],"%Y%m%dT%H%M%S") - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                            newtimeend = (dt.datetime.strptime(newtimestamp.split("_")[1],"%Y%m%dT%H%M%S") - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                            done = False
                            for i in range(nfiles_o2):
                                secondsend = tend_stamp_hr[i]*3600.0 + tend_stamp_min[i]*60 + tend_stamp_sec[i]
                                secondsstart = tzero_stamp_hr[i]*3600.0 + tzero_stamp_min[i]*60 + tzero_stamp_sec[i]
                                tend = (datenow + dt.timedelta(seconds=secondsend) - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                                tstart = (datenow + dt.timedelta(seconds=secondsstart) - dt.datetime(1985,1,1,0,0,0)).total_seconds()   
                                #if( (done==False ) and (math.isclose(tend,newtimeend,abs_tol=1.1)) and math.isclose(tstart,newtimestart,abs_tol=1.1)  ):
    
                                if( (done==False ) and (math.isclose(tend,newtimeend,abs_tol=1.1))):
                                    done = True
                                    fname = 'MethaneAIR_L1B_O2_'+str(tzero_stamp[i])+'_'+str(tend_stamp[i])+'_'+str(tproc_stamp[i])+'_correction_factors_akaze.txt'
                                    data = np.genfromtxt(os.path.join(l1_abs_akaze_DataDir,fname),delimiter=',')
                                    abs_sc_lon=float(data[0])
                                    abs_sc_lat=float(data[1])
                                    abs_off_lon=float(data[2])
                                    abs_off_lat=float(data[3])
                                    abs_rlon=float(data[4])
                                    abs_rlat=float(data[5])
                                    avionics_only = False
                                elif( (done==False ) and (math.isclose(tstart,newtimestart,abs_tol=1.1))):
                                    done = True
                                    fname = 'MethaneAIR_L1B_O2_'+str(tzero_stamp[i])+'_'+str(tend_stamp[i])+'_'+str(tproc_stamp[i])+'_correction_factors_akaze.txt'
                                    data = np.genfromtxt(os.path.join(l1_abs_akaze_DataDir,fname),delimiter=',')
                                    abs_sc_lon=float(data[0])
                                    abs_sc_lat=float(data[1])
                                    abs_off_lon=float(data[2])
                                    abs_off_lat=float(data[3])
                                    abs_rlon=float(data[4])
                                    abs_rlat=float(data[5])
                                    avionics_only = False
                                elif( (done==False ) and (newtimestart<tend) and (nenewtimestart>tstart)  ):
                                    done = True
                                    fname = 'MethaneAIR_L1B_O2_'+str(tzero_stamp[i])+'_'+str(tend_stamp[i])+'_'+str(tproc_stamp[i])+'_correction_factors_akaze.txt'
                                    data = np.genfromtxt(os.path.join(l1_abs_akaze_DataDir,fname),delimiter=',')
                                    abs_sc_lon=float(data[0])
                                    abs_sc_lat=float(data[1])
                                    abs_off_lon=float(data[2])
                                    abs_off_lat=float(data[3])
                                    abs_rlon=float(data[4])
                                    abs_rlat=float(data[5])
                                    avionics_only = False
                        else:
                            done = False
                            try:
                                print('Trying to copy failed GEOAkze nc4 file to failed directory')
                                gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_Fail_DataDir,'MethaneAIR_L1B_CH4_'+timestamp+'.nc'))
                            except:
                                print('Failed to copy GEOAkze nc4 file to failed directory - continuing')
    
                    except:
                        done = False
                        try:
                            print('Trying to copy failed GEOAkze nc4 file to failed directory')
                            gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_Fail_DataDir,'MethaneAIR_L1B_CH4_'+timestamp+'.nc'))
                        except:
                            print('Failed to copy GEOAkze nc4 file to failed directory - continuing')
                        rval_lat = 0
                        rval_lon = 0
                        avionics_only = True
    
                    if(done == False):
                        print('Second Additional O2 file found, but akaze failed or no O2 params found')
                        rval_lat = 0
                        rval_lon = 0
                        avionics_only = True
                    if(   (abs(rval_lon-1) >= 0.05) or (abs(rval_lat-1) >= 0.05) ):
                        print('Second Akaze values fail')
                        rval_lat = 0
                        rval_lon = 0
                        avionics_only = True
                    elif(  (abs(rval_lon-1) < 0.05) and (abs(rval_lat-1) < 0.05)): 
                        cfile = 'MethaneAIR_L1B_CH4_'+str(timestamp)+'_correction_factors_akaze.txt'
                        os.system('cp '+cfile+' '+ l1_rel_akaze_DataDir)
                    else:
                        rval_lat = 0
                        rval_lon = 0
                        avionics_only = True
            else:
                cfile = 'MethaneAIR_L1B_CH4_'+str(timestamp)+'_correction_factors_akaze.txt'
                os.system('cp '+cfile+' '+ l1_rel_akaze_DataDir)
               
            try:
                print('O2 AKAZE correction file = ',fname)
            except:
                pass
            
            #########################################################################################
            if(done == False):
                rval_lat = 0
                rval_lon = 0
                avionics_only = True
                print('No absolute correction parameters matched')
        #########################################################################################
        #########################################################################################
        nframes = granule.nFrame
        #########################################################################################
        #SPLIT THE GRANULES INTO 10 SECOND INTERVALS 
        granule=None
        granuleList = m.F_cut_granule(akaze_ch4_file,granuleSeconds=10)
        #########################################################################################
        #########################################################################################
        
        for i in range(len(granuleList)):
            timestamp = np.min(granuleList[i].frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                                      +np.max(granuleList[i].frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                                      +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
            m.save_cut_avionics(timestamp,granuleList[i],headerStr='MethaneAIR_L1B_CH4_')
            filename = os.path.join(cwd,'MethaneAIR_L1B_CH4_'+timestamp+'.nc') 
            #############################################################  
            # nframe0 and nframe1 are the start and end indices of where to grab avionics lon/lat data from. 
            if(i==0):
                nframe0 = 0 
                nframe1 = granuleList[i].nFrame 
            elif(i!=0):
                if(i==1):
                    nframe0=granuleList[i-1].nFrame
                else:
                    nframe0=0
                    for j in range(i):
                        nframe0 = granuleList[j].nFrame + nframe0 
                nframe1 = granuleList[i].nFrame+nframe0 
            if(avionics_only == False): 
                if(computer == 'Odyssey'):
                    args1 =str(rel_sc_lon)
                    args2 =str(rel_sc_lat )
                    args3 =str(rel_off_lon)
                    args4 =str(rel_off_lat )
                    args5 =str(abs_sc_lon)
                    args6 =str(abs_sc_lat )
                    args7 =str(abs_off_lon)
                    args8 =str(abs_off_lat )
        
                    args9 =str(os.path.join(root_data,flight_nc_file))
                    args10=str(dem_file)
                    args11=str(filename)
                    args12=str(savepath)
                    args13=str(os.path.join(root_data,eop_file))
                    args14=str(os.path.join(root_data,'0User_Functions'))
                    args15=str("LATC")
                    args16=str("LONC")
                    args17=str("PITCH")
                    args18=str("ROLL")
                    args19=str("THDG")
                    args20=str("GEOPTH")
                    args21=str("GGEOIDHT")
                    subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Optimized_NC_Fast_CH4_Hydra.R'), args1, args2, args3, args4, args5, args6,args7, args8, args9, args10, args11, args12, args13,args14,args15,args16,args17,args18,args19,args20,args21])
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
                    args10=str(dem_file)
                    args11=str(filename)
                    args12=str(savepath)
                    args13=str(os.path.join(root_data,eop_file))
                    args14=str(os.path.join(root_data,'0User_Functions'))
                    args15=str("LATC")
                    args16=str("LONC")
                    args17=str("PITCH")
                    args18=str("ROLL")
                    args19=str("THDG")
                    args20=str("GEOPTH")
                    args21=str("GGEOIDHT")
                    subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Optimized_NC_Fast_CH4_Hydra.R'), args1, args2, args3, args4, args5, args6,args7, args8, args9, args10, args11, args12, args13,args14,args15,args16,args17,args18,args19,args20,args21])
            x = m.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(l1DataDir),xtrk_aggfac=1,atrk_aggfac=1,ortho_step='avionics',nframe0=nframe0,nframe1=nframe1,avfile=akaze_ch4_file,av_only = avionics_only)
        
            if(avionics_only == False):
                avair_only = False
            elif(avionics_only != False):
                avair_only = None
    
            lockname=logfile_native+'.lock'
            with FileLock(lockname):
                f = open(logfile_native,'a+') 
                f.write(str(os.path.join(l1DataDir,'MethaneAIR_L1B_CH4_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                f.close()
            os.remove(os.path.join(cwd,'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'))
            name = 'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'
            filename = os.path.join(l1DataDir,name)
            aggregate.main(filename,l1AggDataDir,xfac,yfac,av=avair_only)
            lockname=logfile_agg+'.lock'
            with FileLock(lockname):
                f = open(logfile_agg,'a+') 
                f.write(str(os.path.join(l1AggDataDir,'MethaneAIR_L1B_CH4_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                f.close()
            name = 'MethaneAIR_L1B_CH4'+'_'+timestamp+'.nc'
            filename = os.path.join(l1DataDir,name)
            aggregate.main(filename,l1AggDataDir2,5,1,av=avair_only)
            lockname=logfile_agg2+'.lock'
            with FileLock(lockname):
                f = open(logfile_agg2,'a+') 
                f.write(str(os.path.join(l1AggDataDir2,'MethaneAIR_L1B_CH4_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                f.close()
        
        os.system('rm '+str(akaze_ch4_file)) 
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
        xtrackaggfactor=fitxfac)
    
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
    
    
        mintime = np.min(granule.frameDateTime)
        maxtime = np.max(granule.frameDateTime)
    
        FOV=33.7
        buff = 15000
        dem_file = dem_maker.main(os.path.join(root_data,flight_nc_file),mintime,maxtime,datenow,FOV,buff) 
    
        filename = os.path.join(cwd,'MethaneAIR_L1B_O2_'+timestamp+'.nc')
        #########################################################################################
        # SAVE ENTIRE .NC GRANULE
        #########################################################################################
        o.F_save_L1B_mat(timestamp,granule,headerStr='MethaneAIR_L1B_O2_')
        #########################################################################################
        #LOAD GEOLOCATION FUNCTION IN R
        #########################################################################################
        if(flight!='RF10'):                             
            # If computer is Hydra
            if(computer == 'Hydra'):
                args1=str(os.path.join(root_data,flight_nc_file))
                args2=str(dem_file)
                args3=str(filename)
                args4=str(savepath)
                args5=str(os.path.join(root_data,'0Inputs/EOP-Last5Years_celestrak_20210723.txt'))
                args6=str(os.path.join(root_data,'0User_Functions/'))
                args7=str("LATC")
                args8=str("LONC")
                args9=str("PITCH")
                args10=str("ROLL")
                args11=str("THDG")
                args12=str("GEOPTH")
                args13=str("GGEOIDHT")
                subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Avionics_NC_Fast_Hydra.R'), args1, args2, args3, args4, args5, args6,args7,args8,args9,args10,args11,args12,args13])
            else:
                args1=str(os.path.join(root_data,flight_nc_file))
                args2=str(dem_file)
                args3=str(filename)
                args4=str(savepath)
                args5=str(os.path.join(root_data,'0Inputs/EOP-Last5Years_celestrak_20210723.txt'))
                args6=str(os.path.join(root_data,'0User_Functions/'))
                args7=str("LATC")
                args8=str("LONC")
                args9=str("PITCH")
                args10=str("ROLL")
                args11=str("THDG")
                args12=str("GEOPTH")
                args13=str("GGEOIDHT")
                subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Avionics_NC_Fast_Hydra.R'), args1, args2, args3, args4, args5, args6,args7,args8,args9,args10,args11,args12,args13])
            #########################################################################################
            # WRITE SPLAT FORMATTED L1B FILE - ENTIRE GRANULE 
            #########################################################################################
            # need to save the avionics only full-granule to a temporary scratch directpry for ch4 cross-alignment
            if not os.path.exists(os.path.join(cwd,'AvionicsGranule')):
                os.makedirs(os.path.join(cwd,'AvionicsGranule'))
            avionics_dir = os.path.join(cwd,'AvionicsGranule')

            x = o.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(avionics_dir),1,1,'avionics',nframe0=None,nframe1=None,avfile=None)
            os.remove(os.path.join(cwd,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))
            os.system('cp '+str(os.path.join(avionics_dir,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))+' '+ str(o2dir)) 
            akaze_o2_file = os.path.join(o2dir,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc') 

            """
            name = 'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'
            filename = os.path.join(avionics_dir,name)
            aggregate.main(filename,o2diragg,xfac,yfac,av=True)


            name = 'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'
            filename = os.path.join(avionics_dir,name)
            aggregate.main(filename,o2diragg2,5,1,av=True)


            lockname=logfile_avionics_native+'.lock'
            with FileLock(lockname):
                f = open(logfile_avionics_native,'a+') 
                f.write(str(os.path.join(o2dir,'MethaneAIR_L1B_O2_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                f.close()
            lockname=logfile_avionics_agg+'.lock'
            with FileLock(lockname):
                f = open(logfile_avionics_agg,'a+') 
                f.write(str(os.path.join(o2diragg,'MethaneAIR_L1B_O2_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                f.close()
            lockname=logfile_avionics_agg2+'.lock'
            with FileLock(lockname):
                f = open(logfile_avionics_agg2,'a+') 
                f.write(str(os.path.join(o2diragg2,'MethaneAIR_L1B_O2_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                f.close()
            """

            #########################################################################################
            # NEED TO CALL AKAZE ALGORITHM HERE
            files=[]
            files.append(akaze_o2_file) 
            try:
                gkobj = GEOAkaze(files, file_landsat, 0.0001, 0, 3, 10000,msi_clim_fld = MSI_Climatology, is_histeq=True, is_destriping=False, bandindex_slave=1, w1=0, w2=160)
                gkobj.readslave()
                gkobj.readmaster()
                gkobj.append_master()
                gkobj.akaze()


                abs_sc_lat=gkobj.slope_lat
                abs_sc_lon=gkobj.slope_lon
                abs_off_lat=gkobj.intercept_lat
                abs_off_lon=gkobj.intercept_lon
                rval_lat =gkobj.success
                rval_lon = gkobj.success
                if(gkobj.success==1):
                    gkobj.savetokmz(os.path.join(os.getcwd(),'MethaneAIR_L1B_O2_'+timestamp+'.kmz'))
                    gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_DataDir,'MethaneAIR_L1B_O2_'+timestamp+'.nc'))
                    gkobj.savetotxt(os.path.join(os.getcwd(),'MethaneAIR_L1B_O2_'+timestamp))
                    os.system('cp *.kmz* '+ l1_GEOakaze_kmz_DataDir)
                else:
                    try:
                        print('Trying to copy failed GEOAkze nc4 file to failed directory')
                        gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_Fail_DataDir,'MethaneAIR_L1B_O2_'+timestamp+'.nc'))
                    except:
                        print('Failed to copy GEOAkze nc4 file to failed directory - continuing')

                afile = 'MethaneAIR_L1B_O2_'+str(timestamp)+'_correction_factors_akaze.txt' 
                #########################################################################################
                rel_off_lon = np.float(0.0) 
                rel_sc_lon  = np.float(1.0) 
                rel_off_lat = np.float(0.0) 
                rel_sc_lat  = np.float(1.0)

                abs_off_lon = np.float(abs_off_lon) 
                abs_sc_lon  = np.float(abs_sc_lon ) 
                abs_off_lat = np.float(abs_off_lat) 
                abs_sc_lat  = np.float(abs_sc_lat )
            except:
                try:
                    print('Trying to copy failed GEOAkze nc4 file to failed directory')
                    gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_Fail_DataDir,'MethaneAIR_L1B_O2_'+timestamp+'.nc'))
                except:
                    print('Failed to copy GEOAkze nc4 file to failed directory - continuing')
                rel_off_lon = np.float(0.0) 
                rel_sc_lon  = np.float(1.0) 
                rel_off_lat = np.float(0.0) 
                rel_sc_lat  = np.float(1.0)
                rval_lat = 0.0
                rval_lon = 0.0


            if((abs(rval_lon - 1) >= 0.05 ) or (abs(rval_lat - 1) >= 0.05 )   ):
                print('Akaze has failed with one granule - trying to add more frames')

                #date_end_before = start of current    
                # Searching for granule before this one
                tstart_now = np.min(granule.frameDateTime)
                tend_now = np.max(granule.frameDateTime)
                tstart_now_seconds = (tstart_now - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                tend_now_seconds = (tend_now - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                matched = False  
                for file in os.listdir(o2dir):
                    if('.nc' in file):
                        tnew = file.split('MethaneAIR_L1B_O2_')[1]
                        tnew = tnew.split('.nc')[0]  
                        tend = tnew.split('_')[1]
                        tend = dt.datetime.strptime(tend,'%Y%m%dT%H%M%S')#.strftime('%Y%m%dT%H%M%S')
                        tend = (tend - dt.datetime(1985,1,1,0,0,0)).total_seconds()
                        tstart = tnew.split('_')[0]
                        tstart = dt.datetime.strptime(tstart,'%Y%m%dT%H%M%S')#.strftime('%Y%m%dT%H%M%S')
                        tstart = (tstart - dt.datetime(1985,1,1,0,0,0)).total_seconds()

                        if( (matched == False) and (math.isclose(tend,tstart_now_seconds,abs_tol=2)) and (math.isclose(tstart,tstart_now_seconds,abs_tol=1)!=True)  ):
                            files.append(os.path.join(o2dir,file)) 
                            matchedo2file = file
                            matched = True
                        elif( (matched == False) and (math.isclose(tstart,tend_now_seconds,abs_tol=2) ) and (math.isclose(tend,tend_now_seconds,abs_tol=1)!=True)):
                            files.append(os.path.join(o2dir,file)) 
                            matched = True
                            matchedo2file = file
                if(matched == True):
                    print('Second O2 granule found and added to akaze procedure')
                    try:
                        gkobj = GEOAkaze(files, file_landsat, 0.0001, 0, 3, 10000,msi_clim_fld = MSI_Climatology, is_histeq=True, is_destriping=False, bandindex_slave=1, w1=0, w2=160)
                        gkobj.readslave()
                        gkobj.readmaster()
                        gkobj.append_master()
                        gkobj.akaze()


                        abs_sc_lat=gkobj.slope_lat
                        abs_sc_lon=gkobj.slope_lon
                        abs_off_lat=gkobj.intercept_lat
                        abs_off_lon=gkobj.intercept_lon
                        rval_lat =gkobj.success
                        rval_lon = gkobj.success
                        if(gkobj.success==1):
                            gkobj.savetokmz(os.path.join(os.getcwd(),'MethaneAIR_L1B_O2_'+timestamp+'.kmz'))
                            gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_DataDir,'MethaneAIR_L1B_O2_'+timestamp+'.nc'))
                            gkobj.savetotxt(os.path.join(os.getcwd(),'MethaneAIR_L1B_O2_'+timestamp))
                            os.system('cp *.kmz* '+ l1_GEOakaze_kmz_DataDir)
                        else:
                            try:
                                print('Trying to copy failed GEOAkze nc4 file to failed directory')
                                gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_Fail_DataDir,'MethaneAIR_L1B_O2_'+timestamp+'.nc'))
                            except:
                                print('Failed to copy GEOAkze nc4 file to failed directory - continuing')
                        afile = 'MethaneAIR_L1B_O2_'+str(timestamp)+'_correction_factors_akaze.txt'
                        abs_off_lon = np.float(abs_off_lon) 
                        abs_sc_lon  = np.float(abs_sc_lon ) 
                        abs_off_lat = np.float(abs_off_lat) 
                        abs_sc_lat  = np.float(abs_sc_lat )
                        print("Rvalue Lon = ",rval_lon)
                        print("Rvalue Lat = ",rval_lat)
                    except:
                        try:
                            print('Trying to copy failed GEOAkze nc4 file to failed directory')
                            gkobj.write_to_nc(os.path.join(l1_GEOakaze_nc_Fail_DataDir,'MethaneAIR_L1B_O2_'+timestamp+'.nc'))
                        except:
                            print('Failed to copy GEOAkze nc4 file to failed directory - continuing')
                        avionics_only = True
                else:
                    print('No second O2 granule found for akaze - exitting')
                    avionics_only = True

            else:
                afile = 'MethaneAIR_L1B_O2_'+str(timestamp)+'_correction_factors_akaze.txt'
                #os.system('cp '+afile+' '+ l1_abs_akaze_DataDir)

            if((abs(rval_lon - 1) >= 0.05 ) or (abs(rval_lat - 1) >= 0.05 )   ):
                avionics_only = True
            elif((abs(rval_lon - 1) < 0.05 ) and (abs(rval_lat - 1) < 0.05 )  ):
                avionics_only = False
                os.remove(os.path.join(savepath,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))
                os.system('cp '+afile+' '+ l1_abs_akaze_DataDir)
            else:
                avionics_only = True
            #########################################################################################
            nframes = granule.nFrame
            #########################################################################################
            #SPLIT THE GRANULES INTO 10 SECOND INTERVALS 
            #########################################################################################
            granule=None
            granuleList = o.F_cut_granule(akaze_o2_file,granuleSeconds=10)

            #########################################################################################
            #########################################################################################
            # LOAD OPTIMIZED ORTHORECTIFICATION R CODE
            #########################################################################################

            for i in range(len(granuleList)):
                timestamp = np.min(granuleList[i].frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                                          +np.max(granuleList[i].frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'\
                                          +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
                o.save_cut_avionics(timestamp,granuleList[i],headerStr='MethaneAIR_L1B_O2_')
                filename = os.path.join(cwd,'MethaneAIR_L1B_O2_'+timestamp+'.nc') 
                if(i==0):
                    nframe0 = 0 
                    nframe1 = granuleList[i].nFrame 
                elif(i!=0):
                    if(i==1):
                        nframe0=granuleList[i-1].nFrame
                    else:
                        nframe0=0
                        for j in range(i):
                            nframe0 = granuleList[j].nFrame + nframe0 
                    nframe1 = granuleList[i].nFrame+nframe0 
                #############################################################
                # Avionics Optimized Routine
                #############################################################
                if(avionics_only == False): 
                    if(computer == 'Odyssey'):
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
                        args10=str(dem_file)
                        args11=str(filename)
                        args12=str(savepath)
                        args13=str(os.path.join(root_data,eop_file))
                        args14=str(os.path.join(root_data,'0User_Functions'))
                        args15=str("LATC")
                        args16=str("LONC")
                        args17=str("PITCH")
                        args18=str("ROLL")
                        args19=str("THDG")
                        args20=str("GEOPTH")
                        args21=str("GGEOIDHT")
                        subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Optimized_NC_Fast_O2_Hydra.R'), args1, args2, args3, args4, args5, args6,args7, args8, args9, args10, args11, args12, args13,args14,args15,args16,args17,args18,args19,args20,args21])
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
                        args10=str(dem_file)
                        args11=str(filename)
                        args12=str(savepath)
                        args13=str(os.path.join(root_data,eop_file))
                        args14=str(os.path.join(root_data,'0User_Functions'))
                        args15=str("LATC")
                        args16=str("LONC")
                        args17=str("PITCH")
                        args18=str("ROLL")
                        args19=str("THDG")
                        args20=str("GEOPTH")
                        args21=str("GGEOIDHT")
                        subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Optimized_NC_Fast_O2_Hydra.R'), args1, args2, args3, args4, args5, args6,args7, args8, args9, args10, args11, args12, args13,args14,args15,args16,args17,args18,args19,args20,args21])


                x = o.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(l1DataDir),xtrk_aggfac=1,atrk_aggfac=1,ortho_step='avionics',nframe0=nframe0,nframe1=nframe1,avfile=akaze_o2_file,av_only = avionics_only)

                if(avionics_only == False):
                    avair_only = False
                elif(avionics_only != False):
                    avair_only = None

                lockname=logfile_native+'.lock'
                with FileLock(lockname):
                    f = open(logfile_native,'a+') 
                    f.write(str(os.path.join(l1DataDir,'MethaneAIR_L1B_O2_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                    f.close()
                os.remove(os.path.join(cwd,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))
                name = 'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'
                filename = os.path.join(l1DataDir,name)
                aggregate.main(filename,l1AggDataDir,xfac,yfac,av=avair_only)
                lockname=logfile_agg+'.lock'
                with FileLock(lockname):
                    f = open(logfile_agg,'a+') 
                    f.write(str(os.path.join(l1AggDataDir,'MethaneAIR_L1B_O2_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                    f.close()
                name = 'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'
                filename = os.path.join(l1DataDir,name)
                aggregate.main(filename,l1AggDataDir2,5,1,av=avair_only)
                lockname=logfile_agg2+'.lock'
                with FileLock(lockname):
                    f = open(logfile_agg2,'a+') 
                    f.write(str(os.path.join(l1AggDataDir2,'MethaneAIR_L1B_O2_'+str(timestamp)+'.nc'))+' '+str(priority)+'\n' )
                    f.close()
            os.system('rm '+str(akaze_o2_file)) 
            print('Total Time(s) for ',nframes,' Frames = ' ,(time.time() - t_start))
        else:
            args1=str(os.path.join(root_data,flight_nc_file))
            args2=str(dem_file)
            args3=str(filename)
            args4=str(savepath)
            args5=str(os.path.join(root_data,'0Inputs/EOP-Last5Years_celestrak_20210723.txt'))
            args6=str(os.path.join(root_data,'0User_Functions/'))
            args7=str("LATC")
            args8=str("LONC")
            args9=str("PITCH")
            args10=str("ROLL")
            args11=str("THDG")
            args12=str("GEOPTH")
            args13=str("GGEOIDHT")
            args14=str(celestial_sphere_height_km)
            subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Avionics_NC_Airglow_Hydra.R'), args1, args2, args3, args4, args5, args6,args7,args8,args9,args10,args11,args12,args13,args14])
            #########################################################################################
            # WRITE SPLAT FORMATTED L1B FILE - ENTIRE GRANULE 
            #########################################################################################
            x = o.write_splat_l1_coadd(str(cwd),str(savepath),str(timestamp),str(o2dir),xtrk_aggfac=1,atrk_aggfac=1,ortho_step='avionics')
            os.remove(os.path.join(savepath,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))

            ##########
            # 60 KM 
            ##########
            celestial_sphere_height_km = 60
            # If computer is Hydra
            args1=str(os.path.join(root_data,flight_nc_file))
            args2=str(dem_file)
            args3=str(filename)
            args4=str(savepath)
            args5=str(os.path.join(root_data,'0Inputs/EOP-Last5Years_celestrak_20210723.txt'))
            args6=str(os.path.join(root_data,'0User_Functions/'))
            args7=str("LATC")
            args8=str("LONC")
            args9=str("PITCH")
            args10=str("ROLL")
            args11=str("THDG")
            args12=str("GEOPTH")
            args13=str("GGEOIDHT")
            args14=str(celestial_sphere_height_km)
            subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Avionics_NC_Airglow_Hydra.R'), args1, args2, args3, args4, args5, args6,args7,args8,args9,args10,args11,args12,args13,args14])
            
            #########################################################################################
            # WRITE SPLAT FORMATTED L1B FILE - ENTIRE GRANULE 
            #########################################################################################

            ncfile = nc4.Dataset(os.path.join(o2dir,'MethaneAIR_L1B_O2_'+timestamp+'.nc'),'a',format="NETCDF4")
            ncgroup = ncfile.createGroup('60KM')
            sza = ncgroup.createVariable('SolarZenithAngle',np.float32,('y','x'))
            vza = ncgroup.createVariable('ViewingZenithAngle',np.float32,('y','x'))
            saa = ncgroup.createVariable('SolarAzimuthAngle',np.float32,('y','x'))
            vaa = ncgroup.createVariable('ViewingAzimuthAngle',np.float32,('y','x'))
            aza = ncgroup.createVariable('RelativeAzimuthAngle',np.float32,('y','x'))
            lon = ncgroup.createVariable('Longitude',np.float32,('y','x'))
            lat = ncgroup.createVariable('Latitude',np.float32,('y','x'))
            obsalt = ncgroup.createVariable('ObservationAltitude',np.float32,('y'))
            aclon  = ncgroup.createVariable('AircraftLongitude',np.float32,('y'))
            aclat  = ncgroup.createVariable('AircraftLatitude',np.float32,('y'))
            acaltsurf = ncgroup.createVariable('AircraftAltitudeAboveSurface',np.float32,('y'))
            acsurfalt = ncgroup.createVariable('AircraftSurfaceAltitude',np.float32,('y'))
            acbore = ncgroup.createVariable('AircraftPixelBore',np.float32,('eci','y','x'))
            acpos = ncgroup.createVariable('AircraftPos',np.float32,('eci','y'))
            clon = ncgroup.createVariable('CornerLongitude',np.float32,('c','y','x'))
            clat = ncgroup.createVariable('CornerLatitude',np.float32,('c','y','x'))

            xf=os.listdir(savepath)
            geofile=[]
            for i in range(len(xf)):
                geofile.append(os.path.join(savepath,xf[i]))
            geofile=np.array(geofile)
            if(len(geofile)>1 ):
                print('error: geofile contains more than one file')
                exit()
            else:
                df = nc4.Dataset(geofile[0]).variables['SolarZenithAngle'][:,:]
                aza_temp = np.zeros((df.shape))
                df=None
                vaa_temp = nc4.Dataset(geofile[0]).variables['ViewingAzimuthAngle'][:,:]
                saa_temp = nc4.Dataset(geofile[0]).variables['SolarAzimuthAngle'][:,:]
                # Compute VLIDORT relative azimuth (180 from usual def)
                aza_temp = vaa_temp - (saa_temp + 180.0)
                idv = np.isfinite(aza_temp)
                if( (aza_temp[idv] < 0.0).any() or (aza_temp[idv] > 360.0).any() ):
                    aza_temp[aza_temp<0.0] = aza_temp[aza_temp<0.0]+360.0
                    aza_temp[aza_temp>360.0] = aza_temp[aza_temp>360.0]-360.0

                sza[:,:] = (nc4.Dataset(geofile[0]).variables['SolarZenithAngle'][:,:])#.transpose(1,0)
                vza[:,:] = (nc4.Dataset(geofile[0]).variables['ViewingZenithAngle'][:,:])#.transpose(1,0)
                saa[:,:] = (nc4.Dataset(geofile[0]).variables['SolarAzimuthAngle'][:,:])#.transpose(1,0)
                vaa[:,:] = (nc4.Dataset(geofile[0]).variables['ViewingAzimuthAngle'][:,:])#.transpose(1,0)
                aza[:,:] = aza_temp
                lon[:,:] = (nc4.Dataset(geofile[0]).variables['Longitude'][:,:])#.transpose(1,0)
                lat[:,:] = (nc4.Dataset(geofile[0]).variables['Latitude'][:,:])#.transpose(1,0)
                obsalt[:] = nc4.Dataset(geofile[0]).variables['AircraftAltitudeAboveWGS84'][:]*1e-3
                aclon[:]  = nc4.Dataset(geofile[0]).variables['AircraftLongitude'][:]
                aclat[:]  = nc4.Dataset(geofile[0]).variables['AircraftLatitude'][:]
                acaltsurf[:] = nc4.Dataset(geofile[0]).variables['AircraftAltitudeAboveSurface'][:]*1e-3
                acsurfalt[:] = nc4.Dataset(geofile[0]).variables['AircraftSurfaceAltitude'][:]*1e-3
                acbore[:,:,:] = (nc4.Dataset(geofile[0]).variables['AircraftPixelBore'][:,:,:])#.transpose((2,1,0)) 
                acpos[:,:] = (nc4.Dataset(geofile[0]).variables['AircraftPos'][:,:])#.transpose(1,0)
                clon[:,:,:] = (nc4.Dataset(geofile[0]).variables['CornerLongitude'][:,:,:])#.transpose((2,1,0))
                clat[:,:,:] = (nc4.Dataset(geofile[0]).variables['CornerLatitude'][:,:,:])#.transpose((2,1,0))

                ncfile.close()
            os.remove(os.path.join(savepath,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))
                             
            ##########
            # 70 KM 
            ##########
            celestial_sphere_height_km = 70
            # If computer is Hydra
            args1=str(os.path.join(root_data,flight_nc_file))
            args2=str(dem_file)
            args3=str(filename)
            args4=str(savepath)
            args5=str(os.path.join(root_data,'0Inputs/EOP-Last5Years_celestrak_20210723.txt'))
            args6=str(os.path.join(root_data,'0User_Functions/'))
            args7=str("LATC")
            args8=str("LONC")
            args9=str("PITCH")
            args10=str("ROLL")
            args11=str("THDG")
            args12=str("GEOPTH")
            args13=str("GGEOIDHT")
            args14=str(celestial_sphere_height_km)
            subprocess.call(['Rscript',os.path.join(os.path.dirname(__file__),'Orthorectification_Avionics_NC_Airglow_Hydra.R'), args1, args2, args3, args4, args5, args6,args7,args8,args9,args10,args11,args12,args13,args14])
            #########################################################################################
            # WRITE SPLAT FORMATTED L1B FILE - ENTIRE GRANULE 
            #########################################################################################

            ncfile = nc4.Dataset(os.path.join(o2dir,'MethaneAIR_L1B_O2_'+timestamp+'.nc'),'a',format="NETCDF4")
            ncgroup = ncfile.createGroup('70KM')
            sza = ncgroup.createVariable('SolarZenithAngle',np.float32,('y','x'))
            vza = ncgroup.createVariable('ViewingZenithAngle',np.float32,('y','x'))
            saa = ncgroup.createVariable('SolarAzimuthAngle',np.float32,('y','x'))
            vaa = ncgroup.createVariable('ViewingAzimuthAngle',np.float32,('y','x'))
            aza = ncgroup.createVariable('RelativeAzimuthAngle',np.float32,('y','x'))
            lon = ncgroup.createVariable('Longitude',np.float32,('y','x'))
            lat = ncgroup.createVariable('Latitude',np.float32,('y','x'))
            obsalt = ncgroup.createVariable('ObservationAltitude',np.float32,('y'))
            aclon  = ncgroup.createVariable('AircraftLongitude',np.float32,('y'))
            aclat  = ncgroup.createVariable('AircraftLatitude',np.float32,('y'))
            acaltsurf = ncgroup.createVariable('AircraftAltitudeAboveSurface',np.float32,('y'))
            acsurfalt = ncgroup.createVariable('AircraftSurfaceAltitude',np.float32,('y'))
            acbore = ncgroup.createVariable('AircraftPixelBore',np.float32,('eci','y','x'))
            acpos = ncgroup.createVariable('AircraftPos',np.float32,('eci','y'))
            clon = ncgroup.createVariable('CornerLongitude',np.float32,('c','y','x'))
            clat = ncgroup.createVariable('CornerLatitude',np.float32,('c','y','x'))
 
            xf=os.listdir(savepath)
            geofile=[]
            for i in range(len(xf)):
                geofile.append(os.path.join(savepath,xf[i]))
            geofile=np.array(geofile)
            if(len(geofile)>1 ):
                print('error: geofile contains more than one file')
                exit()
            else:
                df = nc4.Dataset(geofile[0]).variables['SolarZenithAngle'][:,:]
                aza_temp = np.zeros((df.shape))
                df=None
                vaa_temp = nc4.Dataset(geofile[0]).variables['ViewingAzimuthAngle'][:,:]
                saa_temp = nc4.Dataset(geofile[0]).variables['SolarAzimuthAngle'][:,:]
                # Compute VLIDORT relative azimuth (180 from usual def)
                aza_temp = vaa_temp - (saa_temp + 180.0)
                idv = np.isfinite(aza_temp)
                if( (aza_temp[idv] < 0.0).any() or (aza_temp[idv] > 360.0).any() ):
                    aza_temp[aza_temp<0.0] = aza_temp[aza_temp<0.0]+360.0
                    aza_temp[aza_temp>360.0] = aza_temp[aza_temp>360.0]-360.0

                sza[:,:] = (nc4.Dataset(geofile[0]).variables['SolarZenithAngle'][:,:])#.transpose(1,0)
                vza[:,:] = (nc4.Dataset(geofile[0]).variables['ViewingZenithAngle'][:,:])#.transpose(1,0)
                saa[:,:] = (nc4.Dataset(geofile[0]).variables['SolarAzimuthAngle'][:,:])#.transpose(1,0)
                vaa[:,:] = (nc4.Dataset(geofile[0]).variables['ViewingAzimuthAngle'][:,:])#.transpose(1,0)
                aza[:,:] = aza_temp
                lon[:,:] = (nc4.Dataset(geofile[0]).variables['Longitude'][:,:])#.transpose(1,0)
                lat[:,:] = (nc4.Dataset(geofile[0]).variables['Latitude'][:,:])#.transpose(1,0)
                obsalt[:] = nc4.Dataset(geofile[0]).variables['AircraftAltitudeAboveWGS84'][:]*1e-3
                aclon[:]  = nc4.Dataset(geofile[0]).variables['AircraftLongitude'][:]
                aclat[:]  = nc4.Dataset(geofile[0]).variables['AircraftLatitude'][:]
                acaltsurf[:] = nc4.Dataset(geofile[0]).variables['AircraftAltitudeAboveSurface'][:]*1e-3
                acsurfalt[:] = nc4.Dataset(geofile[0]).variables['AircraftSurfaceAltitude'][:]*1e-3
                acbore[:,:,:] = (nc4.Dataset(geofile[0]).variables['AircraftPixelBore'][:,:,:])#.transpose((2,1,0)) 
                acpos[:,:] = (nc4.Dataset(geofile[0]).variables['AircraftPos'][:,:])#.transpose(1,0)
                clon[:,:,:] = (nc4.Dataset(geofile[0]).variables['CornerLongitude'][:,:,:])#.transpose((2,1,0))
                clat[:,:,:] = (nc4.Dataset(geofile[0]).variables['CornerLatitude'][:,:,:])#.transpose((2,1,0))

                ncfile.close()
            os.remove(os.path.join(savepath,'MethaneAIR_L1B_O2'+'_'+timestamp+'.nc'))
            print('Total Time(s) for ',nframes,' Frames = ' ,(time.time() - t_start))
                             
                             
if __name__ == "__main__":
    main()
