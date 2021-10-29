import os
from filelock import FileLock
import numpy as np
import netCDF4 as nc4
from tqdm import tqdm
import v4_align_ch4_single
import argparse


def main(input_dir: str):
    files = []
    direct = os.path.join(input_dir,'O2_15x3/') 
    align_dir = os.path.join(input_dir,'O2_15x3_Aligned/') 
    logfile = os.path.join(input_dir,'O2_15x3_Aligned/log_file_o2_aligned.txt')
    
    native_logfile = os.path.join(input_dir,'O2_NATIVE/log_file.txt')
    g = open(native_logfile,'r')
    native = g.readlines()
    g.close()
    nfiles = len(native)
    native_o2_files = []
    native_o2_priority = []
    for i in range(nfiles):
        native_o2_files.append(os.path.basename(native[i].split(' ')[0]))
        native_o2_priority.append((native[i].split(' ')[1]).split('\n')[0])
    
    
    for file in os.listdir(direct):
        if file.endswith(".nc"):
            files.append(file)
    
    
    ch4_start = 9 
    ch4_end = 65
    
    nfiles=len(files)
    
    
    for i in range(nfiles):
        if(os.path.exists(os.path.join(align_dir,files[i])) == False):
            found_match=False
            for j in range(len(native_o2_files)):
                if(found_match==False and (str(os.path.basename(files[i]) == str(native_o2_files[j]))) ):
                    priority =native_o2_priority[j]      
                    found_match=True
            
            lockname=logfile+'.lock'
            with FileLock(lockname):
                f = open(logfile,'a+') 
                f.write(str(os.path.join(align_dir,files[i]))+' '+str(priority)+'\n' )
                f.close()
            os.system('cp '+str(os.path.join(direct,files[i]))+' '+str(align_dir) )
     
            df = nc4.Dataset(os.path.join(align_dir,files[i]),'a')
                  
            df.groups['Band1'].variables['Radiance'][:,:,0:ch4_start]  = np.nan
            df.groups['Band1'].variables['RadianceUncertainty'][:,:,0:ch4_start]  = np.nan
            df.groups['Band1'].variables['Wavelength'][:,:,0:ch4_start]  = np.nan
            
            df.groups['Band1'].variables['Radiance'][:,:,ch4_end:]  = np.nan
            df.groups['Band1'].variables['RadianceUncertainty'][:,:,ch4_end:]  = np.nan
            df.groups['Band1'].variables['Wavelength'][:,:,ch4_end:]  = np.nan
    
    
    ####### 5x1 #########
    
    files = []
    direct = os.path.join(input_dir,'O2_5x1/') 
    align_dir = os.path.join(input_dir,'O2_5x1_Aligned/') 
    logfile = os.path.join(input_dir,'O2_5x1_Aligned/log_file_o2_aligned.txt')
    
    native_logfile = os.path.join(input_dir,'O2_NATIVE/log_file.txt')
    g = open(native_logfile,'r')
    native = g.readlines()
    g.close()
    nfiles = len(native)
    native_o2_files = []
    native_o2_priority = []
    for i in range(nfiles):
        native_o2_files.append(os.path.basename(native[i].split(' ')[0]))
        native_o2_priority.append((native[i].split(' ')[1]).split('\n')[0])
    
    
    for file in os.listdir(direct):
        if file.endswith(".nc"):
            files.append(file)
    
    
    ch4_start = 29 
    ch4_end = 197
    
    nfiles=len(files)
    
    
    for i in range(nfiles):
        if(os.path.exists(os.path.join(align_dir,files[i])) == False):
    
            found_match=False
            for j in range(len(native_o2_files)):
                if(found_match==False and (str(os.path.basename(files[i]) == str(native_o2_files[j]))) ):
                    priority =native_o2_priority[j]      
                    found_match=True
            
            lockname=logfile+'.lock'
            with FileLock(lockname):
                f = open(logfile,'a+') 
                f.write(str(os.path.join(align_dir,files[i]))+' '+str(priority)+'\n' )
                f.close()
    
            os.system('cp '+str(os.path.join(direct,files[i]))+' '+str(align_dir) )
     
            df = nc4.Dataset(os.path.join(align_dir,files[i]),'a')
     
            df.groups['Band1'].variables['Radiance'][:,:,0:ch4_start]  = np.nan
            df.groups['Band1'].variables['RadianceUncertainty'][:,:,0:ch4_start]  = np.nan
            df.groups['Band1'].variables['Wavelength'][:,:,0:ch4_start]  = np.nan
            
            df.groups['Band1'].variables['Radiance'][:,:,ch4_end:]  = np.nan
            df.groups['Band1'].variables['RadianceUncertainty'][:,:,ch4_end:]  = np.nan
            df.groups['Band1'].variables['Wavelength'][:,:,ch4_end:]  = np.nan

if __name__ == "__main__":
    # argparse is a python standard library for parsing command line input args
    parser = argparse.ArgumentParser(description="Align")
    parser.add_argument("--input-dir", required=True, help="Input flight directory")
    args = parser.parse_args()
    
    # call align func with the user specified paths
    align(args.input_dir) 
