#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt
import struct
import os
import logging
from scipy.io import loadmat
from netCDF4 import Dataset
import netCDF4 as nc4
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import time
from sys import exit
import numba as nb


"""
Created on Wed Jun  9 14:12:26 2021

@author: conwayek
"""
def F_isrf_convolve_fft(w1,s1,w2,isrf_w,isrf_dw0,isrf_lut0,ISRFtype,fitisrf):
    """
    astropy.covolute.convolve_fft-based convolution using wavelength-dependent isrf
    w1:
        high-resolution wavelength
    s1:
        high-resolution spectrum
    w2:
        low-resolution wavelength
    isrf_w:
        center wavelength grid of isrf lut
    isrf_dw:
        wavelength grid on which isrfs are defined
    isrf_lut:
        instrument spectral response function look up table
    """
    from astropy.convolution import convolve_fft
    from scipy.interpolate import RegularGridInterpolator
    from scipy import interpolate,optimize

    if(fitisrf == False):
        if isrf_lut0.shape != (len(isrf_w),len(isrf_dw0)):
            raise ValueError('isrf_lut dimension incompatible!')
            return np.full(w2.shape,np.nan)
        # make sure w1 and isrf_dw have the same resolution
        w1_step = np.median(np.diff(w1))
        isrf_dw_min = np.min(isrf_dw0)
        isrf_dw_max = -isrf_dw_min
        isrf_dw = np.linspace(isrf_dw_min,isrf_dw_max,int((isrf_dw_max-isrf_dw_min)/w1_step)+1)
        isrf_lut = np.zeros((len(isrf_w),len(isrf_dw)))
        for (iw,w) in enumerate(isrf_w):
            if(ISRFtype == 'ISRF'):
                interp_func = interpolate.interp1d(isrf_dw0,isrf_lut0[iw,:])
                isrf_lut[iw,:] = interp_func(isrf_dw)
            elif(ISRFtype == 'GAUSS'):
                popt, pcov = optimize.curve_fit(gaussian, isrf_lut0[iw,:])
                isrf_lut[iw,:] = gaussian(isrf_dw,popt[0])
            elif(ISRFtype == 'SUPER'):
                popt, pcov = optimize.curve_fit(supergaussian, isrf_dw0, isrf_lut0[iw,:])
                isrf_lut[iw,:] = supergaussian(isrf_dw,popt[0],popt[1])
            else:
                print('ISRF Function Not Coded: ONLY ISRF, GAUSS and SUPER Allowed')
                exit()
        # note that the isrf is flipped: convolution is the mirror-image of kernel averaging
        s2_fft_lut = np.array([convolve_fft(s1,isrf_lut[iw,::-1]) for (iw,w) in enumerate(isrf_w)])
        inter_func = RegularGridInterpolator((isrf_w,w1),s2_fft_lut,bounds_error=False)
        
        return inter_func((w2,w2))
    elif( (fitisrf == True) and ISRFtype == 'SQUEEZE'):
        if isrf_lut0.shape != (len(isrf_w),len(isrf_dw0[0,:])):
            raise ValueError('isrf_lut dimension incompatible!')
            return np.full(w2.shape,np.nan)
        # make sure w1 and isrf_dw have the same resolution
        
        s2_fft_lut = np.zeros(( len(isrf_w),len(s1) ))

        for (iw,w) in enumerate(isrf_w):
            w1_step = np.median(np.diff(w1))
            isrf_dw_min = np.min(isrf_dw0[iw,:])
            isrf_dw_max = -isrf_dw_min
            isrf_dw = np.linspace(isrf_dw_min,isrf_dw_max,int((isrf_dw_max-isrf_dw_min)/w1_step)+1)
            isrf_lut = np.zeros((len(isrf_w),len(isrf_dw)))
            interp_func = interpolate.interp1d(isrf_dw0[iw,:],isrf_lut0[iw,:])
            isrf_lut[iw,:] = interp_func(isrf_dw)
            s2_fft_lut[iw,:] = np.array([convolve_fft(s1,isrf_lut[iw,::-1])])
        inter_func = RegularGridInterpolator((isrf_w,w1),s2_fft_lut,bounds_error=False)
        return inter_func((w2,w2))
    
 
            
    elif( (fitisrf == True) and ISRFtype == 'SUPER'):
        if (isrf_lut0.shape) != (2,len(isrf_w)):
            raise ValueError('isrf_lut dimension incompatible!')
            return np.full(w2.shape,np.nan)
        # make sure w1 and isrf_dw have the same resolution
        w1_step = np.median(np.diff(w1))
        isrf_dw_min = np.min(isrf_dw0)
        isrf_dw_max = -isrf_dw_min
        isrf_dw = np.linspace(isrf_dw_min,isrf_dw_max,int((isrf_dw_max-isrf_dw_min)/w1_step)+1)
        isrf_lut = np.zeros((len(isrf_w),len(isrf_dw)))
        for (iw,w) in enumerate(isrf_w):
            isrf_lut[iw,:] = supergaussian(isrf_dw,isrf_lut0[0,iw],isrf_lut0[1,iw])

        # note that the isrf is flipped: convolution is the mirror-image of kernel averaging
        s2_fft_lut = np.array([convolve_fft(s1,isrf_lut[iw,::-1]) for (iw,w) in enumerate(isrf_w)])
        inter_func = RegularGridInterpolator((isrf_w,w1),s2_fft_lut,bounds_error=False)
        return inter_func((w2,w2))   

    
            
    else:
        #IN THIS CASE, ISRF_LUT0 ARE THE FITTED HWHM: NO NEED TO FIT THE DATA
        if len(isrf_lut0) != (len(isrf_w)):
            raise ValueError('isrf_lut dimension incompatible!')
            return np.full(w2.shape,np.nan)
        # make sure w1 and isrf_dw have the same resolution
        w1_step = np.median(np.diff(w1))
        isrf_dw_min = np.min(isrf_dw0)
        isrf_dw_max = -isrf_dw_min
        isrf_dw = np.linspace(isrf_dw_min,isrf_dw_max,int((isrf_dw_max-isrf_dw_min)/w1_step)+1)
        isrf_lut = np.zeros((len(isrf_w),len(isrf_dw)))
        for (iw,w) in enumerate(isrf_w):
            isrf_lut[iw,:] = gaussian(isrf_dw,isrf_lut0[iw])

        # note that the isrf is flipped: convolution is the mirror-image of kernel averaging
        s2_fft_lut = np.array([convolve_fft(s1,isrf_lut[iw,::-1]) for (iw,w) in enumerate(isrf_w)])
        inter_func = RegularGridInterpolator((isrf_w,w1),s2_fft_lut,bounds_error=False)
        
        return inter_func((w2,w2))        


#SUPER GAUSSIAN ISRF
def supergaussian(x, w, k):     
    from math import gamma
    awk = k/( 2.0 * w * gamma(1/k)  )
    return awk * np.exp(-(abs(x/w) )**k)
     

#GAUSSIAN ISRF
def gaussian(x,width):
    return (1.0/(width*np.sqrt(2*np.pi) ) )  * np.exp(-0.5 * (x**2) / (width**2) )    
   
def FWHM(x,y):
    hmx = half_max_x(x,y)
    fwhm = hmx[1] - hmx[0]
    return fwhm
    
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
    
def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


def fitspectra(Data,l1FitPath,whichBand,solarRefFile,calidata,o2spectraReffile,ciaspectraReffile,co2spectraReffile,h2ospectraReffile,ch4spectraReffile,fitSZA,SZA,ALT,l1DataDir,frameDateTime,pixlimitX,
               isrf_lut,isrf_w,isrf_dw0,wavelength,pixlimit,fitisrf,ISRF,xtol,ftol,xtrackaggfactor):
    
    from scipy import interpolate,optimize
    from lmfit import minimize, Parameters, Parameter,fit_report
    from scipy.interpolate import RegularGridInterpolator
    import math
    from skimage.measure import block_reduce
    
    

    #NUMBER OF FRAMES CAN BE USED TO GET FIRST GUESS ON THE VALUE OF SOLAR SCALING
    numframes = int(Data.shape[2])
    

    radiance = np.nanmean(Data,axis=2)
    radiance = radiance*1e-14



    #aggregate the data in the xtrack dimenssion
    norm_2d = block_reduce(np.ones(radiance.shape), block_size=(xtrackaggfactor, 1),func=np.mean )
    # Valid pixels
    valpix = np.zeros(radiance.shape)
    idv = np.logical_and(np.isfinite(radiance),radiance>0.0)
    valpix[idv] = 1.0
    valpix_agg = block_reduce(valpix, block_size=(xtrackaggfactor,1),func=np.mean )
    valpix_agg = valpix_agg / norm_2d

    # Coadd radiance data
    rad_obs = block_reduce(radiance,block_size=(xtrackaggfactor,1),func=np.mean)
    rad_obs = rad_obs / norm_2d

    # Coadd wvl data
    wavelength = block_reduce(wavelength,block_size=(xtrackaggfactor,1),func=np.mean)
    wavelength = wavelength / norm_2d
    
    
    # aggregate ISRF data in the same way
    norm_3d = block_reduce(np.ones(isrf_lut.shape), block_size=(xtrackaggfactor,1,1),func=np.mean )
    # Valid pixels
    valpix = np.zeros(isrf_lut.shape)
    idv = np.logical_and(np.isfinite(isrf_lut),isrf_lut>0.0)
    valpix[idv] = 1.0
    valpix_agg = block_reduce(valpix, block_size=(xtrackaggfactor,1,1),func=np.mean )
    valpix_agg = valpix_agg / norm_3d

    # Coadd radiance data
    isrf_lut = block_reduce(isrf_lut,block_size=(xtrackaggfactor,1,1),func=np.mean)
    isrf_lut = isrf_lut / norm_3d    


    
    ##############################
    #SOLAR DATA
    ncid = Dataset(solarRefFile, 'r')
    if(whichBand == 'CH4'):
        f = (ncid['Wavelength'][:] > 1590) & (ncid['Wavelength'][:] < 1700)
        refWavelength = ncid['Wavelength'][:][f].data
        refSolarIrradiance = ncid['SSI'][:][f].data
    else:
        f = (ncid['Wavelength'][:] > 1245) & (ncid['Wavelength'][:] < 1310)
        refWavelength = ncid['Wavelength'][:][f].data
        refSolarIrradiance = ncid['SSI'][:][f].data
    ncid.close()
    ##############################
    
    
    ##############################
    # CROSS SECTIONAL DATA
    #UT XSECTIONS + US ATM FILE        
    #US ST. ATM. DATA
    usatmospath = os.path.join(calidata,str('AFGLUS_atmos.txt'))
    uspressure = np.loadtxt(usatmospath,usecols=0)
    usheight = np.loadtxt(usatmospath,usecols=1)
    ustemperature = np.loadtxt(usatmospath,usecols=2)
    usair = np.loadtxt(usatmospath,usecols=3)
    usCH4 = np.loadtxt(usatmospath,usecols=4)
    usCO2 = np.loadtxt(usatmospath,usecols=5)
    usH2O = np.loadtxt(usatmospath,usecols=6)
    
    
    specwave = refWavelength
    if(whichBand == 'O2'):

        #ncid = Dataset(o2spectraReffile, 'r')
        specwave = np.arange(refWavelength[0],refWavelength[-1],0.001)
        h2opath = os.path.join(calidata,str('h2o_lut_HITRAN2020_5e-3cm-1.nc'))
        o2path = os.path.join(calidata,'o2_lut_HIT2020_5e-3cm-1.nc')
        ###################################################
        new = Dataset(h2opath, 'r')
        TH2O   = new.variables['CrossSection'][:,:,:].data  
        PressH2O = new.variables['Pressure'][:].data
        WvlH2O = new.variables['Wavelength'][:].data
        TempH2O   = new.variables['Temperature'][:].data 
        
        WvlH2O = 1.0e7/WvlH2O
        WvlH2O = np.flip(WvlH2O)
        TH2O = np.flip(TH2O,axis=2)
        
        new.close()
        ###################################################
        new = Dataset(o2path, 'r')
        Temp = new.variables['Temperature'][:].data
        Press = new.variables['Pressure'][:].data
        Wvl = new.variables['Wavelength'][:].data
        TO2   = new.variables['CrossSection'][:,:,:].data  
        
        
        Wvl = 1.0e7/Wvl
        Wvl = np.flip(Wvl)
        TO2 = np.flip(TO2,axis=2)
        
        
        new.close()
        ##########
        
        file = os.path.join(calidata,str('O2_CIA_296K_all.nc'))
        data = Dataset(file,'r')
        WvlCIA = data['Wavelength'][:]
        TCIA = data['XSection'][:]
        data.close()
        
        
        fnO2 = RegularGridInterpolator((Press,Temp,Wvl), TO2)
        fnH2O = RegularGridInterpolator((PressH2O,TempH2O,WvlH2O), TH2O)


        wgtO2 = np.zeros(len(Wvl))
        wgtH2O = np.zeros(len(WvlH2O))
        wgtCIA = np.zeros(len(WvlCIA))
                    
        columntotalO2 = 0.0
        columntotalH2O = 0.0
        columntotalCIA = 0.0
        xO2 = fnO2((400,250,Wvl))
        for j in range(len(Wvl)):
            wgtO2[j] = wgtO2[j] + xO2[j]
        for i in range( len(usheight) ):
            if(usheight[i] <= ALT):
                """
                xO2 = fnO2((uspressure[i],ustemperature[i],Wvl)) *  abs(1.0/np.cos(SZA) + 1)
                for j in range(len(Wvl)):
                    wgtO2[j] = wgtO2[j] + xO2[j]
                columntotalO2 = columntotalO2 + usair[i]
                """
                xH2O = fnH2O((uspressure[i],ustemperature[i],WvlH2O)) * usH2O[i] *  abs(1.0/np.cos(SZA) + 1.0)
                columntotalH2O = columntotalH2O + usH2O[i] 
                
                
                #xCIA = fnCIA(ustemperature[i],WvlCIA) *  abs(1.0/np.cos(SZA) + 1.0)
                #for j in range(len(WvlCIA)):
                    #wgtCIA[j] = wgtCIA[j] + xCIA[j]
                for j in range(len(WvlH2O)):    
                    wgtH2O[j] = wgtH2O[j] + xH2O[j]
                
            else:
                """
                xO2 = fnO2((uspressure[i],ustemperature[i],Wvl)) *  abs(1.0/np.cos(SZA) )
                for i in range(len(Wvl)):
                    wgtO2[i] = wgtO2[i] + xO2[i]
                columntotalO2 = columntotalO2 + usair[i]
                """
                xH2O = fnH2O((uspressure[i],ustemperature[i],WvlH2O)) * usH2O[i] * abs(1.0/np.cos(SZA))
                columntotalH2O = columntotalH2O + usH2O[i]
                
                #xCIA = fnCIA(ustemperature[i],WvlCIA) *  abs(1.0/np.cos(SZA) )
                #for j in range(len(WvlCIA)):
                    #wgtCIA[j] = wgtCIA[j] + xCIA[j]
                for j in range(len(WvlH2O)): 
                    wgtH2O[j] = wgtH2O[j] + xH2O[j]
            
        
        wgtH2O = wgtH2O/columntotalH2O

        wgtCIA = TCIA#wgtCIA
        
        wgtO2 = wgtO2

        yH2O = interpolate.interp1d(WvlH2O,wgtH2O)
        yO2 = interpolate.interp1d(Wvl,wgtO2)
        yCIA = interpolate.interp1d(WvlCIA,wgtCIA)

        
        H2O = np.zeros(len(specwave))
        O2 = np.zeros(len(specwave))
        CIA = np.zeros(len(specwave))

        for i in range(len(specwave)):
            H2O[i] = yH2O(specwave[i])
            O2[i] = yO2(specwave[i])
            CIA[i] = yCIA(specwave[i])




        #o2path = os.path.join(calidata,spectraReffile) 
        ##o2path = os.path.join(calidata,str('o2_lut_1200-1750nm_0p02fwhm_1e21vcd_mesat.nc'))
        #new = Dataset(o2path, 'r')
        #TO2   = new.variables['CrossSection'][:,:,:].data  
        #Temp = new.variables['Temperature'][:].data
        #Press = new.variables['Pressure'][:].data
        #Wvl = new.variables['Wavelength'][:].data
        #
        #
        #fn = RegularGridInterpolator((Press,Temp,Wvl), TO2)
        #
        #wgtO2 = np.zeros(len(Wvl))
        #columntotal = 0.0
        #for i in range( len(usheight) ):
        #    if(usheight[i] <= 58.0):
        #        x = fn((uspressure[i],ustemperature[i],Wvl)) * usair[i]
        #        wgtO2 = wgtO2 + x
        #        columntotal = columntotal + usair[i]
        #wgtO2 = wgtO2/columntotal
        #
       ## wgtO2 = fn((1013,288,Wvl)) 
        #y = interpolate.interp1d(Wvl,wgtO2)
        #O2 = np.zeros(len(specwave))
        #for i in range(len(specwave)):
        #    O2[i] = y(specwave[i])


        #CIA  = ncid.variables['CIAO2AIR'][:]
        ##CIA  = ncid.variables['CIATCCON'][:]
        #ncid.close()

        
    else:
        h2opath = os.path.join(calidata,str('h2o_lut_HITRAN2020_5e-3cm-1.nc'))
        co2path = os.path.join(calidata,str('co2_lut_1200-1750nm_0p02fwhm_1e21vcd_mesat.nc'))
        ch4path = os.path.join(calidata,str('ch4_lut_HITRAN2020_5e-3cm-1_g0_update.nc'))
        
        new = Dataset(h2opath, 'r')
        TH2O   = new.variables['CrossSection'][:,:,:].data  
        TempH2O = new.variables['Temperature'][:].data
        PressH2O = new.variables['Pressure'][:].data
        WvlH2O = new.variables['Wavelength'][:].data

        WvlH2O = 1.0e7/WvlH2O
        WvlH2O = np.flip(WvlH2O)
        TH2O = np.flip(TH2O,axis=2)

        new.close()

        new = Dataset(ch4path, 'r')
        TCH4 = new.variables['CrossSection'][:,:,:].data 
        TempCH4 = new.variables['Temperature'][:].data
        PressCH4 = new.variables['Pressure'][:].data
        WvlCH4 = new.variables['Wavelength'][:].data
        WvlCH4 = 1.0e7/WvlCH4
        WvlCH4 = np.flip(WvlCH4)
        TCH4 = np.flip(TCH4,axis=2)
        new.close()
        
        new = Dataset(co2path, 'r')
        TCO2 = new.variables['CrossSection'][:,:,:].data  
        TempCO2 = new.variables['Temperature'][:].data
        PressCO2 = new.variables['Pressure'][:].data
        WvlCO2 = new.variables['Wavelength'][:].data
        new.close()
        
        fnCH4 = RegularGridInterpolator((PressCH4,TempCH4,WvlCH4), TCH4)
        fnCO2 = RegularGridInterpolator((PressCO2,TempCO2,WvlCO2), TCO2)
        fnH2O = RegularGridInterpolator((PressH2O,TempH2O,WvlH2O), TH2O)
        
        wgtCH4 = np.zeros(len(WvlCH4))
        wgtCO2 = np.zeros(len(WvlCO2))
        wgtH2O = np.zeros(len(WvlH2O))
                    
        columntotalCH4 = 0.0
        columntotalCO2 = 0.0
        columntotalH2O = 0.0
        
        if(fitSZA):
            
            for i in range( len(usheight) ):
                if(usheight[i] <= ALT):
                    xCH4 = fnCH4((uspressure[i],ustemperature[i],WvlCH4)) * usCH4[i] * (1.0/np.cos(SZA) + 1.0)
                    wgtCH4 = wgtCH4 + xCH4
                    columntotalCH4 = columntotalCH4 + usCH4[i]
                    ###
                    xCO2 = fnCO2((uspressure[i],ustemperature[i],WvlCO2)) * usCO2[i] * (1.0/np.cos(SZA) + 1.0)
                    wgtCO2 = wgtCO2 + xCO2
                    columntotalCO2 = columntotalCO2 + usCO2[i]
                    ###
                    xH2O = fnH2O((uspressure[i],ustemperature[i],WvlH2O)) * usH2O[i] * (1.0/np.cos(SZA) + 1.0)
                    wgtH2O = wgtH2O + xH2O
                    columntotalH2O = columntotalH2O + usH2O[i]
                else:
                    xCH4 = fnCH4((uspressure[i],ustemperature[i],WvlCH4)) * usCH4[i] * (1.0/np.cos(SZA) )
                    wgtCH4 = wgtCH4 + xCH4
                    columntotalCH4 = columntotalCH4 + usCH4[i]
                    ###
                    xCO2 = fnCO2((uspressure[i],ustemperature[i],WvlCO2)) * usCO2[i] * (1.0/np.cos(SZA))
                    wgtCO2 = wgtCO2 + xCO2
                    columntotalCO2 = columntotalCO2 + usCO2[i]
                    ###
                    xH2O = fnH2O((uspressure[i],ustemperature[i],WvlH2O)) * usH2O[i] * (1.0/np.cos(SZA))
                    wgtH2O = wgtH2O + xH2O
                    columntotalH2O = columntotalH2O + usH2O[i]
                
            
            
        else:
            for i in range( len(usheight) ):
                if(usheight[i] <= 58.0):
                    xCH4 = fnCH4((uspressure[i],ustemperature[i],WvlCH4)) * usCH4[i]
                    wgtCH4 = wgtCH4 + xCH4
                    columntotalCH4 = columntotalCH4 + usCH4[i]
                    ###
                    xCO2 = fnCO2((uspressure[i],ustemperature[i],WvlCO2)) * usCO2[i]
                    wgtCO2 = wgtCO2 + xCO2
                    columntotalCO2 = columntotalCO2 + usCO2[i]
                    ###
                    xH2O = fnH2O((uspressure[i],ustemperature[i],WvlH2O)) * usH2O[i]
                    wgtH2O = wgtH2O + xH2O
                    columntotalH2O = columntotalH2O + usH2O[i]
                
                
                
                
                
        wgtCO2 = wgtCO2/columntotalCO2
        wgtCH4 = wgtCH4/columntotalCH4
        wgtH2O = wgtH2O/columntotalH2O
        
        yCH4 = interpolate.interp1d(WvlCH4,wgtCH4)
        CH4 = np.zeros(len(specwave))
        
        yCO2 = interpolate.interp1d(WvlCO2,wgtCO2)
        CO2 = np.zeros(len(specwave))
        
        yH2O = interpolate.interp1d(WvlH2O,wgtH2O)
        H2O = np.zeros(len(specwave))
        
        for i in range(len(specwave)):
            CH4[i] = yCH4(specwave[i])
            CO2[i] = yCO2(specwave[i])
            H2O[i] = yH2O(specwave[i])
    """        
    plt.plot(specwave,CH4)
    plt.savefig('ch4_start.png')
    plt.close()            
        
    plt.plot(specwave,CO2)
    plt.savefig('co2_start.png')
    plt.close()            

    plt.plot(specwave,H2O)
    plt.savefig('h2o_start.png')
    plt.close()            
    """
    ##############################
    
    # DEFINE CROSS TRACK POSITIONS TO FIT
    counter = 0
    
    if(whichBand == 'CH4'): 
        headerStr='MethaneAIR_L1B_CH4_'
        l1FitISRFPath = os.path.join(l1DataDir,headerStr
                                +np.min(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                +np.max(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                +str('ISRF_proxy_fit_CH4.txt'))

        f = open(l1FitPath,"w")
        g = open(l1FitISRFPath,"w")
    else:
        headerStr='MethaneAIR_L1B_O2_'
        l1FitISRFPath = os.path.join(l1DataDir,headerStr
                                +np.min(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                +np.max(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                +str('ISRF_proxy_fit_O2.txt'))
        f = open(l1FitPath,"w")
        g = open(l1FitISRFPath,"w")


    #THE FITTING STARTS HERE
    start = 0

    #for j in range (pixlimitX[0],pixlimitX[1]):
    wavelength_raw = wavelength 
    for j in range(rad_obs.shape[0]):
    
        xTrack = j
        error=False
        if (whichBand == 'CH4'):
            isrf_lut0 = isrf_lut[xTrack,:,:]
        else:
            isrf_lut0 = isrf_lut[xTrack,:,:]
          #  isrf = np.transpose(isrf_lut, [2,1,0] )
          #  isrf_lut0 = isrf[xTrack,:,:] 
            
        #OBSERVED DATA   
        wavelength_obs = wavelength_raw[xTrack,:].squeeze()
        radiance_obs = rad_obs[xTrack,:].squeeze()
        

        if math.isnan( wavelength_obs[pixlimit[0]]):
            print(str(xTrack)+ " has a nan values in wvl window")
        elif math.isnan( radiance_obs[pixlimit[0]]):
            print(str(xTrack)+ " has a nan values in rad window")
        elif math.isnan( wavelength_obs[pixlimit[1]]):
            print(str(xTrack)+ " has a nan values in wvl window")
        elif math.isnan( radiance_obs[pixlimit[1]]):
            print(str(xTrack)+ " has a nan values in rad window")
        else:    
            print('FITTING TRACK: '+str(xTrack))
            
            # GET THE GRID OF INTEREST FOR INTERPOLATION
            gridwidth = 1*(pixlimit[1] - pixlimit[0])
            d = np.linspace(wavelength_obs[pixlimit[0]],wavelength_obs[pixlimit[1]], gridwidth+1  )
            
            ##############################
            # DEFINE FITTING VARIABLES:
            ##############################
            params = Parameters()
            #BASELINE CORRECTION
            
            if(start == 0):
            

                if(whichBand == 'CH4'):
                    f0 = Parameter('par_f'+str(0),0.1402161187)
                    params.add(f0)
                    sc_solar = Parameter('scale_solar',0.00573476*numframes,min=0)
                    params.add(sc_solar) 
                    f1 = Parameter('baseline0',-0.12)
                    f2 = Parameter('baseline1',8.1097e-05)
                    params.add(f1)
                    params.add(f2)

                else:
                    f0 = Parameter('par_f'+str(0),0.149264)
                    params.add(f0)
                    sc_solar = Parameter('scale_solar',0.0153476*numframes,min=0)
                    params.add(sc_solar) 
                    f1 = Parameter('baseline0', -1.34526326)
                    f2 = Parameter('baseline1',0.00106887)
                    #f3 = Parameter('baseline2',-6.3130e-04)
                    params.add(f1)
                    params.add(f2)
                    #params.add(f3)
                    #b1 = Parameter('al0',1.0)
                  #  b2 = Parameter('al1',0.0)
                  #  b3 = Parameter('al2',0)
                   # params.add(b1)
                  #  params.add(b2)
                  #  params.add(b3)


                if(whichBand == 'CH4'):
                    sc_co2 = Parameter('scale_co2',6.7378e+22,max=1e24, min=1)
                    sc_ch4 = Parameter('scale_ch4', 2.9832e+20,max=1e24, min=1)
                    sc_h2o = Parameter('scale_h2o', 1.5151e+23,max=1e25, min=1)
                    params.add(sc_co2)
                    params.add(sc_ch4)
                    params.add(sc_h2o)
                else:
                    sc_o2 = Parameter('scale_o2',1.13e+25,max=1e28, min=1)
                    params.add(sc_o2)
                    #sc_cia = Parameter('scale_cia',8e+24,max=1e28, min=1)
                    #params.add(sc_cia)
                    
                    
                                    
                if(fitisrf == True):

                    if(ISRF == 'SQUEEZE'):
                       # p1 = Parameter('squeeze',1.0,min=0.6,max=1.3)
                       # params.add(p1)

                        wavelength = np.zeros((len(isrf_dw0),len(isrf_w) ))
                        width = np.ones(len(isrf_w))
                        sharp = np.ones(len(isrf_w))
                        
                        for i in range(len(isrf_w)):
                            wavelength[:,i] = isrf_w[i] + isrf_dw0[:]
                        d_max = np.max(d)
                        d_min = np.min(d)     
                        
                        count=0
                        index = []
                        for i in range(len(isrf_w)):
                            if((d_min < wavelength[-1,i]) and (d_max > wavelength[0,i]) ):
                                index.append(i)
                                p1 = Parameter('squeeze'+str(count),0.9381123,max=1.2,min=0.8)
                                params.add(p1)
                                p2 = Parameter('sharp'+str(count),1.051526,max=1.2,min=0.8)
                                params.add(p2)
                                count=count+1
                            else:
                                pass
                    
                    
                    
                    
                    
                    
                    else:  
                        wavelength = np.zeros((len(isrf_dw0),len(isrf_w) ))
                        width = np.zeros(len(isrf_w))
                        
                        for i in range(len(isrf_w)):
                            wavelength[:,i] = isrf_w[i] + isrf_dw0[:]
                        d_max = np.max(d)
                        d_min = np.min(d)                  
                              
                              
                        if(ISRF == 'GAUSS'):
                            for i in range(len(isrf_w)):
                                popt, pcov = optimize.curve_fit(gaussian,isrf_dw0,isrf_lut0[i,:])
                                width[i] = popt[0]
                            count=0
                            index = []
                            for i in range(len(isrf_w)):
                                if((d_min < wavelength[-1,i]) and (d_max > wavelength[0,i]) ):
                                    index.append(i)
                                    p1 = Parameter('width'+str(count),0.1445,max=1,min=0.01)
                                    params.add(p1)
                                    count=count+1
                                else:
                                    pass
                        elif(ISRF == 'SUPER'):
                            shape = np.zeros(len(isrf_w))
                            for i in range(len(isrf_w)):
                                popt, pcov = optimize.curve_fit(supergaussian,isrf_dw0,isrf_lut0[i,:])
                                width[i] = popt[0]
                                shape[i] = popt[1]
                            isrf_super = np.vstack((width,shape))
                            count=0
                            index = []
                            for i in range(len(isrf_w)):
                                if((d_min < wavelength[-1,i]) and (d_max > wavelength[0,i]) ):
                                    p1 = Parameter('width'+str(count),width[i],max=1,min=0.01)
                                    p2 = Parameter('shape'+str(count),shape[i],max=5,min=0.01)
                                    params.add(p1)
                                    params.add(p2)
                                    count=count+1
                                    index.append(i)
                                else:
                                    pass
                else:
                    index = None   
                #####
                # temporary holders for fit parameters - updated upon sucessful iteration
                f0CH4=0.0
                b0CH4=1.0
                b1CH4=0.0
                solarCH4=1.0
                scaleco2=1e21
                scalech4=1e23
                scaleh2o=1e21
                #######
                f0O2=0.0
                b0O2=1.0
                b1O2=0.0
                b2O2=0.0
                solarO2=1.0
                scaleo2=1e23
                
            else:
                if(whichBand == 'CH4'):
                    f0 = Parameter('par_f'+str(0),f0CH4)
                    params.add(f0)
                    sc_solar = Parameter('scale_solar',solarCH4,min=0)
                    params.add(sc_solar) 
                    f1 = Parameter('baseline0',b0CH4)
                    f2 = Parameter('baseline1',b1CH4)
                    params.add(f1)
                    params.add(f2)
                    
                    sc_co2 = Parameter('scale_co2',scaleco2,max=1e24, min=1)
                    sc_ch4 = Parameter('scale_ch4', scalech4,max=1e24, min=1)
                    sc_h2o = Parameter('scale_h2o', scaleh2o,max=1e25, min=1)
                    params.add(sc_co2)
                    params.add(sc_ch4)
                    params.add(sc_h2o)

                else:
                    f0 = Parameter('par_f'+str(0),f0O2)
                    params.add(f0)
                    sc_solar = Parameter('scale_solar',solarO2,min=0)
                    params.add(sc_solar) 
                    f1 = Parameter('baseline0',b0O2)
                    f2 = Parameter('baseline1',b1O2)
                    #f3 = Parameter('baseline2',b2O2)
                    params.add(f1)
                    params.add(f2)
                    #params.add(f3)
                    
                    sc_o2 = Parameter('scale_o2',scaleo2,max=1e28, min=1)
                    params.add(sc_o2)
                    #sc_cia = Parameter('scale_cia',8e+24,max=1e28, min=1)
                    #params.add(sc_cia)

                    
                    
                                    
                if(fitisrf == True):

                    if(ISRF == 'SQUEEZE'):
                        #p1 = squeeze
                       # params.add(p1)
                        for i in range(len(index)):
                            p1 = Parameter('squeeze'+str(i),width[index[i]],max=1.2,min=0.8)
                            params.add(p1)
                            p2 = Parameter('sharp'+str(i),sharp[index[i]],max=1.2,min=0.8)
                            params.add(p2)


                    if(ISRF == 'GAUSS'):
                        for i in range(len(index)):
                            p1 = Parameter('width'+str(i),width[index[i]],max=1,min=0.01)
                            params.add(p1)

                    elif(ISRF == 'SUPER'):
                        for i in range(len(index)):
                            p1 = Parameter('width'+str(i),width[index[i]],max=1,min=0.01)
                            p2 = Parameter('shape'+str(i),shape[index[i]],max=5,min=0.01)
                            params.add(p1)

                else:
                    index = None

            fit_kws={'xtol':float(xtol),'ftol':float(ftol)}
            ##############################
            # FIT THE DATA
            #first_fit = time.time()
            if(whichBand == 'CH4'):
                lsqFit = minimize( spectrumResidual_CH4, params, args=(specwave,CH4,CO2,H2O,\
                radiance_obs,wavelength_obs,xTrack,pixlimit,fitisrf,ISRF,isrf_lut0,
                isrf_w,isrf_dw0,index,refWavelength,refSolarIrradiance),method='leastsq',max_nfev=1000,**fit_kws)
            else:
                lsqFit = minimize( spectrumResidual_O2, params, args=(specwave,O2,CIA,\
                radiance_obs,wavelength_obs,xTrack,pixlimit,fitisrf,ISRF,isrf_lut0,
                isrf_w,isrf_dw0,index,refWavelength,refSolarIrradiance),method='leastsq',max_nfev=1000,**fit_kws)
            #second_fit = time.time()
            #delta = (second_fit - first_fit)

            #print('time for that fit = ', delta)
            ##############################
                   
            ##############################
            # GET THE WAVELENGTH SHIFTS
            ##############################
            p = np.zeros(1)
            for i in range(0,1):
                p[i] = lsqFit.params['par_f'+str(i)].value
            ##############################

            ##############################
            # PRINT THE RESULTS OF THE FIT
            print(fit_report(lsqFit))    
            ##############################

            if(whichBand == 'O2'):
                if((lsqFit.params['scale_o2'].stderr == None) or (lsqFit.params['scale_solar'].stderr == None) or \
                (lsqFit.params['par_f'+str(0)].stderr == None)  ):
                    error=True
            elif(whichBand == 'CH4'):
                if((lsqFit.params['scale_co2'].stderr == None) or (lsqFit.params['scale_solar'].stderr == None) or \
                 (lsqFit.params['par_f'+str(0)].stderr == None) ):
                     error=True
            else:
                pass
            
            print(error,whichBand) 
            
            
            if(whichBand == 'CH4' and error == False):
                errorch4 = 100*lsqFit.params['scale_ch4'].stderr/lsqFit.params['scale_ch4'].value
                errorco2 = 100*lsqFit.params['scale_co2'].stderr/lsqFit.params['scale_co2'].value
                errorh2o = 100*lsqFit.params['scale_h2o'].stderr/lsqFit.params['scale_h2o'].value
                errorsolar = 100*lsqFit.params['scale_solar'].stderr/lsqFit.params['scale_solar'].value
                errorf0 = 100*lsqFit.params['par_f'+str(0)].stderr/lsqFit.params['par_f'+str(0)].value
                errorb0CH4=100.0*lsqFit.params['baseline0'].stderr/lsqFit.params['baseline0'].value
                errorb1CH4=100.0*lsqFit.params['baseline1'].stderr/lsqFit.params['baseline1'].value
            elif(whichBand == 'O2' and error == False):
                erroro2 = 100*lsqFit.params['scale_o2'].stderr/lsqFit.params['scale_o2'].value
                #errorcia = 100*lsqFit.params['scale_cia'].stderr/lsqFit.params['scale_cia'].value
                errorsolar = 100*lsqFit.params['scale_solar'].stderr/lsqFit.params['scale_solar'].value
                errorf0 = 100*lsqFit.params['par_f'+str(0)].stderr/lsqFit.params['par_f'+str(0)].value
                errorb0o2=100.0*lsqFit.params['baseline0'].stderr/lsqFit.params['baseline0'].value
                errorb1o2=100.0*lsqFit.params['baseline1'].stderr/lsqFit.params['baseline1'].value
                #errorb2o2=100.0*lsqFit.params['baseline2'].stderr/lsqFit.params['baseline2'].value
            else:
                pass
 
            if(fitisrf == 'True'): 
                if(index == None):
                    pass
                elif(index != None):
                    widtherr = np.zeros(len(index))  
                    for i in range(len(index)):
                         if(lsqFit.params['width'+str(i)].stderr == None):
                              pass
                         else:
                              widtherr[i] = 100.0*lsqFit.params['width'+str(i)].stderr/lsqFit.params['width'+str(i)].value
           
            if(whichBand == 'CH4' and error == False):
                 if( (abs(errorch4) >= 100.0) or (abs(errorco2) >= 100.0) or (abs(errorh2o) >= 100.0) or (abs(errorf0) >= 100.0) or (abs(errorsolar) >= 100.0) ):
                      error=True
            elif(whichBand == 'O2' and error == False):
                 if( (abs(erroro2) >= 100.0) or (abs(errorf0) >= 100.0) or (abs(errorsolar) >= 100.0) ):    
                      error=True
            else:
                pass

            # STORE FIT RESULTS
            if( error == True):
                pass
            else:
        
                    counter = counter + 1
                    
                    start = 1
                    
                    if(whichBand == 'CH4'):
                        f0CH4=lsqFit.params['par_f'+str(0)].value
                        solarCH4=lsqFit.params['scale_solar'].value
                        b0CH4=lsqFit.params['baseline0'].value
                        b1CH4=lsqFit.params['baseline1'].value
                        scaleco2=lsqFit.params['scale_co2'].value
                        scalech4=lsqFit.params['scale_ch4'].value
                        scaleh2o=lsqFit.params['scale_h2o'].value
                        
                        f.write(str(int(xtrackaggfactor))+' '+str(int(j))+' '+str(f0CH4)+' '+str(solarCH4)+' '+str(b0CH4)+' '+str(b1CH4)+' '+str(scalech4)+' '+str(scaleco2)+' '+str(scaleh2o)+'\n')
                        f.flush()
                        
                        if(ISRF=='GAUSS'):
                            g.write(str(int(xtrackaggfactor))+' '+str(int(j)))
                            g.flush()
                            for k in range(len(index)):
                                g.write(' '+str(index[k])+' '+str(lsqFit.params['width'+str(k)].value))
                                g.flush()
                            g.write('\n')
                            g.flush()
                        elif(ISRF=='SQUEEZE'):
                            g.write(str(int(xtrackaggfactor))+' '+str(int(j)))
                            g.flush()
                            for k in range(len(index)):
                                g.write(' '+str(index[k])+' '+str(lsqFit.params['squeeze'+str(k)].value)+' '+str(lsqFit.params['sharp'+str(k)].value))
                                g.flush()
                            g.write('\n')
                            g.flush()
                        elif(ISRF=='SUPER'):
                            g.write(str(int(xtrackaggfactor))+' '+str(int(j)))
                            g.flush()
                            for k in range(len(index)):
                                g.write(' '+str(index[k])+' '+str(lsqFit.params['width'+str(k)].value)+' '+str(lsqFit.params['shape'+str(k)].value))
                                g.flush()
                            g.write('\n')
                            g.flush()


                    else:
                        f0O2=lsqFit.params['par_f'+str(0)].value
                        solarO2=lsqFit.params['scale_solar'].value
                        b0O2=lsqFit.params['baseline0'].value
                        b1O2=lsqFit.params['baseline1'].value
                        #b2O2=lsqFit.params['baseline2'].value
                        scaleo2=lsqFit.params['scale_o2'].value
                        #f.write(str(int(xtrackaggfactor))+' '+str(int(j))+' '+str(f0O2)+' '+str(solarO2)+' '+str(b0O2)+' '+str(b1O2)+' '+str(b2O2)+' '+str(scaleo2)+'\n')
                        f.write(str(int(xtrackaggfactor))+' '+str(int(j))+' '+str(f0O2)+' '+str(solarO2)+' '+str(b0O2)+' '+str(b1O2)+' '+str(scaleo2)+'\n')
                        f.flush()
                        if(ISRF=='GAUSS'):
                            g.write(str(int(xtrackaggfactor))+' '+str(int(j)))
                            g.flush()
                            for k in range(len(index)):
                                g.write(' '+str(index[k])+' '+str(lsqFit.params['width'+str(k)].value))
                                g.flush()
                            g.write('\n')
                            g.flush()
                        elif(ISRF=='SQUEEZE'):
                            g.write(str(int(xtrackaggfactor))+' '+str(int(j)))
                            g.flush()
                            for k in range(len(index)):
                                g.write(' '+str(index[k])+' '+str(lsqFit.params['squeeze'+str(k)].value)+' '+str(lsqFit.params['sharp'+str(k)].value))
                                g.flush()
                            g.write('\n')
                            g.flush()
                        elif(ISRF=='SUPER'):
                            g.write(str(int(xtrackaggfactor))+' '+str(int(j)))
                            g.flush()
                            for k in range(len(index)):
                                g.write(' '+str(index[k])+' '+str(lsqFit.params['width'+str(k)].value)+' '+str(lsqFit.params['shape'+str(k)].value))
                                g.flush()
                            g.write('\n')                       
                            g.flush()
                        
    f.close()
    g.close()
    time.sleep(30) 
    return()

###################################################

"""
**************************************************
"""

###################################################
# CH4 CHANNEL FORWARD MODEL
################################################### 

def spectrumResidual_CH4(params,specwave,CH4,CO2,H2O,radiance_obs,wavelength_obs,xTrack,pixlimit,fitisrf,ISRF,isrf_lut0,
                         isrf_w,isrf_dw0,index,refWavelength,refSolarIrradiance):
    from scipy import interpolate,optimize
    # WAVELENGTH GRID OF INTEREST
    gridwidth = pixlimit[1] - pixlimit[0]
    d = np.linspace(wavelength_obs[pixlimit[0]],wavelength_obs[pixlimit[1]], gridwidth+1  )
    # CONVOLVED SPECTRA IN THE INTERVAL D[:]

    a0 = params['baseline0'].value
    a1 = params['baseline1'].value
    
    baseline = np.zeros(gridwidth+1)
    
    for i in range(gridwidth+1):
         baseline[i] = a0 + (a1 * d[i]) 
    

    #NOW WE NEED TO CREATE THE SHIFTED SPECTRUM: 
    newobs_wavelength = np.zeros(gridwidth+1)
    # Set up polynomial coefficients to fit ncol<-->wvl
    p = np.zeros(1)
    for i in range(0,1):
        p[i] = params['par_f'+str(i)].value
        
    # UPDATE OBSERVED WAVELENGTHS/SPECTRA
    radnew = np.zeros(gridwidth+1)
    idx_finite = np.isfinite(radiance_obs)
    radinterp = interpolate.interp1d(wavelength_obs[idx_finite],radiance_obs[idx_finite])  
    Ioriginal = np.zeros(gridwidth+1)
    for i in range(gridwidth+1):
        newobs_wavelength[i] =  d[i] + p[0]
        radnew[i] = radinterp(newobs_wavelength[i])
        Ioriginal[i] = radinterp(d[i])
        
    if(fitisrf == True):
    
    
        if( ISRF == 'SQUEEZE'):
        
            width=np.ones(len(isrf_w))
            sharp=np.ones(len(isrf_w))
            isrf_dw0_new = np.zeros((len(isrf_w),len(isrf_dw0))  )
            #CREATE NEW dw0 grids
            for i in range(len(isrf_w)):
                isrf_dw0_new[i,:] = isrf_dw0
            for i in range(len(index)):
                width[index[i]] = params['squeeze'+str(i)].value
                sharp[index[i]] = params['sharp'+str(i)].value
                
                fwhm0 = FWHM(isrf_dw0,isrf_lut0[index[i],:])                  
                isrf_lut0[index[i],:] = (isrf_lut0[index[i],:])** sharp[index[i]]                    
                fwhm1 = FWHM(isrf_dw0,isrf_lut0[index[i],:])
                stretchfactor = fwhm0/fwhm1   
                isrf_dw0_new[index[i],:] = isrf_dw0 * width[index[i]]*stretchfactor
                
                
            ISOLAR = F_isrf_convolve_fft(refWavelength,refSolarIrradiance,newobs_wavelength,isrf_w,isrf_dw0_new,isrf_lut0,ISRF,fitisrf)
            ICH4 = F_isrf_convolve_fft(specwave,CH4,newobs_wavelength,isrf_w,isrf_dw0_new,isrf_lut0,ISRF,fitisrf)
            ICO2 = F_isrf_convolve_fft(specwave,CO2,newobs_wavelength,isrf_w,isrf_dw0_new,isrf_lut0,ISRF,fitisrf)
            IH2O = F_isrf_convolve_fft(specwave,H2O,newobs_wavelength,isrf_w,isrf_dw0_new,isrf_lut0,ISRF,fitisrf)

            
            
            
            
        elif( ISRF == 'GAUSS'):
            width=np.zeros(len(isrf_w))
            for i in range(len(isrf_w)):
                popt, pcov = optimize.curve_fit(gaussian,isrf_dw0,isrf_lut0[i,:])
                width[i] = popt[0]
            for i in range(len(index)):
                width[index[i]] = params['width'+str(i)].value

            ISOLAR = F_isrf_convolve_fft(refWavelength,refSolarIrradiance,newobs_wavelength,isrf_w,isrf_dw0,width,ISRF,fitisrf)
            ICH4 = F_isrf_convolve_fft(specwave,CH4,newobs_wavelength,isrf_w,isrf_dw0,width,ISRF,fitisrf)
            ICO2 = F_isrf_convolve_fft(specwave,CO2,newobs_wavelength,isrf_w,isrf_dw0,width,ISRF,fitisrf)
            IH2O = F_isrf_convolve_fft(specwave,H2O,newobs_wavelength,isrf_w,isrf_dw0,width,ISRF,fitisrf)
        else:
            shape = np.zeros(len(isrf_w))
            for i in range(len(isrf_w)):
                popt, pcov = optimize.curve_fit(supergaussian,isrf_dw0,isrf_lut0[i,:])
                width[i] = popt[0]
                shape[i] = popt[1]
            isrf_super = np.vstack((width,shape))
            for i in range(len(index)):
                width[index[i]] = params['width'+str(i)].value
                shape[index[i]] = params['shape'+str(i)].value
                #UPDATE ISRF WITH NEW SUERGAUSSIAN PARAMETERS
                isrf_super[0,index[i]] = width[index[i]]
                isrf_super[1,index[i]] = shape[index[i]]
            

            ISOLAR = F_isrf_convolve_fft(refWavelength,refSolarIrradiance,newobs_wavelength,isrf_w,isrf_dw0,isrf_super,ISRF,fitisrf)
            ICH4 = F_isrf_convolve_fft(specwave,CH4,newobs_wavelength,isrf_w,isrf_dw0,isrf_super,ISRF,fitisrf)
            ICO2 = F_isrf_convolve_fft(specwave,CO2,newobs_wavelength,isrf_w,isrf_dw0,isrf_super,ISRF,fitisrf)
            IH2O = F_isrf_convolve_fft(specwave,H2O,newobs_wavelength,isrf_w,isrf_dw0,isrf_super,ISRF,fitisrf)
        
        
        
    else:
        ISOLAR = F_isrf_convolve_fft(refWavelength,refSolarIrradiance,newobs_wavelength,isrf_w,isrf_dw0,isrf_lut0,ISRF,fitisrf)
        ICH4 = F_isrf_convolve_fft(specwave,CH4,newobs_wavelength,isrf_w,isrf_dw0,isrf_lut0,ISRF,fitisrf)
        ICO2 = F_isrf_convolve_fft(specwave,CO2,newobs_wavelength,isrf_w,isrf_dw0,isrf_lut0,ISRF,fitisrf)
        IH2O = F_isrf_convolve_fft(specwave,H2O,newobs_wavelength,isrf_w,isrf_dw0,isrf_lut0,ISRF,fitisrf)

    #InewFIT = interpolate.splrep(newobs_wavelength,radnew)
    #Inew = interpolate.splev(newobs_wavelength,InewFIT,der=0)
    

    #CREATE SIMULATED SPECTRA
    ISOLAR = ISOLAR * params['scale_solar'].value
    ICH4 = np.exp(-ICH4 * params['scale_ch4'].value)
    ICO2 = np.exp(-ICO2 * params['scale_co2'].value )   
    IH2O = np.exp(-IH2O * params['scale_h2o'].value )       

    Isim = np.zeros(gridwidth+1)
    Isim = (ICH4 + ICO2 + IH2O ) * ISOLAR + baseline

    residual = np.zeros(gridwidth + 1)
    residual = Isim - Ioriginal
   
    
    return ( residual)
###################################################

"""
**************************************************
"""

###################################################
# O2 CHANNEL FORWARD MODEL
################################################### 
def spectrumResidual_O2(params,specwave,O2,CIA,radiance_obs,wavelength_obs,xTrack,pixlimit,fitisrf,ISRF,isrf_lut0,
                         isrf_w,isrf_dw0,index,refWavelength,refSolarIrradiance):
   from scipy import interpolate,optimize

   gridwidth = 1*(pixlimit[1] - pixlimit[0])
   d = np.linspace(wavelength_obs[pixlimit[0]],wavelength_obs[pixlimit[1]], gridwidth+1  )
   
   
   a0 = params['baseline0'].value
   a1 = params['baseline1'].value
   #a2 = params['baseline2'].value
   
   
 #  b1 = params['al1'].value
 #  b2 = params['al2'].value
   
   
  # albedo = np.zeros(width+1)
   baseline = np.zeros(gridwidth+1)
   
   for i in range(gridwidth+1):
     #   albedo[i] =  1.0 + (b1 * (d[i] - 1250)) + (b2 * (d[i] - 1250.0)**2) 
        #baseline[i] = a0 + (a1 * d[i]) + (a2*d[i]*d[i]) 
        baseline[i] = a0 + (a1 * d[i]) #+ (a2*d[i]*d[i]) 
        


   #NOW WE NEED TO CREATE THE SHIFTED SPECTRUM: 
   newobs_wavelength = np.zeros(gridwidth+1)
   # Set up polynomial coefficients to fit ncol<-->wvl
   p = np.zeros(1)
   for i in range(0,1):
       p[i] = params['par_f'+str(i)].value
       
   # UPDATE OBSERVED WAVELENGTHS/SPECTRA
   radnew = np.zeros(gridwidth+1)
   idx_finite = np.isfinite(radiance_obs)
   IobsFIT = interpolate.splrep(wavelength_obs[idx_finite],radiance_obs[idx_finite])
   Ioriginal = interpolate.splev(d,IobsFIT,der=0)
   radinterp = interpolate.interp1d(wavelength_obs[idx_finite],radiance_obs[idx_finite])  
   for i in range(gridwidth+1):
       newobs_wavelength[i] =  d[i] + p[0]
   

   radnew = radinterp(newobs_wavelength)
            
   #InewFIT = interpolate.splrep(newobs_wavelength,radnew)
   #Inew = interpolate.splev(newobs_wavelength,InewFIT,der=0)

   if(fitisrf == True):
   
   
       if( ISRF == 'SQUEEZE'):
            width=np.ones(len(isrf_w))
            sharp=np.ones(len(isrf_w))
            isrf_dw0_new = np.zeros((len(isrf_w),len(isrf_dw0))  )
            #CREATE NEW dw0 grids
            for i in range(len(isrf_w)):
                isrf_dw0_new[i,:] = isrf_dw0
                
            for i in range(len(index)):
                width[index[i]] = params['squeeze'+str(i)].value
                sharp[index[i]] = params['sharp'+str(i)].value
                
                fwhm0 = FWHM(isrf_dw0,isrf_lut0[index[i],:])                  
                isrf_lut0[index[i],:] = (isrf_lut0[index[i],:])** sharp[index[i]]                    
                fwhm1 = FWHM(isrf_dw0,isrf_lut0[index[i],:])
                stretchfactor = fwhm0/fwhm1   
                isrf_dw0_new[index[i],:] = isrf_dw0 * width[index[i]]*stretchfactor
                
            ISOLAR = F_isrf_convolve_fft(refWavelength,refSolarIrradiance,newobs_wavelength,isrf_w,isrf_dw0_new,isrf_lut0,ISRF,fitisrf)
            IO2 = F_isrf_convolve_fft(specwave,O2,newobs_wavelength,isrf_w,isrf_dw0_new,isrf_lut0,ISRF,fitisrf)
            ICIA = F_isrf_convolve_fft(specwave,CIA,newobs_wavelength,isrf_w,isrf_dw0_new,isrf_lut0,ISRF,fitisrf)

           
           
       elif( ISRF == 'GAUSS'):
           width=np.zeros(len(isrf_w))
           for i in range(len(isrf_w)):
               popt, pcov = optimize.curve_fit(gaussian,isrf_dw0,isrf_lut0[i,:])
               width[i] = popt[0]
           for i in range(len(index)):
               width[index[i]] = params['width'+str(i)].value

           ISOLAR = F_isrf_convolve_fft(refWavelength,refSolarIrradiance,newobs_wavelength,isrf_w,isrf_dw0,width,ISRF,fitisrf)
           IO2 = F_isrf_convolve_fft(specwave,O2,newobs_wavelength,isrf_w,isrf_dw0,width,ISRF,fitisrf)
           ICIA = F_isrf_convolve_fft(specwave,CIA,newobs_wavelength,isrf_w,isrf_dw0,width,ISRF,fitisrf)
           
       else:
            shape = np.zeros(len(isrf_w))
            for i in range(len(isrf_w)):
                popt, pcov = optimize.curve_fit(supergaussian,isrf_dw0,isrf_lut0[i,:])
                width[i] = popt[0]
                shape[i] = popt[1]
            isrf_super = np.vstack((width,shape))
            for i in range(len(index)):
                width[index[i]] = params['width'+str(i)].value
                shape[index[i]] = params['shape'+str(i)].value
                #UPDATE ISRF WITH NEW SUERGAUSSIAN PARAMETERS
                isrf_super[0,index[i]] = width[index[i]]
                isrf_super[1,index[i]] = shape[index[i]]
            

            ISOLAR = F_isrf_convolve_fft(refWavelength,refSolarIrradiance,newobs_wavelength,isrf_w,isrf_dw0,isrf_super,ISRF,fitisrf)
            ICIA = F_isrf_convolve_fft(specwave,CIA,newobs_wavelength,isrf_w,isrf_dw0,isrf_super,ISRF,fitisrf)
            IO2 = F_isrf_convolve_fft(specwave,O2,newobs_wavelength,isrf_w,isrf_dw0,isrf_super,ISRF,fitisrf)
            

   else:
       ISOLAR = F_isrf_convolve_fft(refWavelength,refSolarIrradiance,newobs_wavelength,isrf_w,isrf_dw0,isrf_lut0,ISRF,fitisrf)
       IO2 = F_isrf_convolve_fft(specwave,O2,newobs_wavelength,isrf_w,isrf_dw0,isrf_lut0,ISRF,fitisrf)
       ICIA = F_isrf_convolve_fft(specwave,CIA,newobs_wavelength,isrf_w,isrf_dw0,isrf_lut0,ISRF,fitisrf)  
       



   #CREATE SIMULATED SPECTRA
   ISOLAR = ISOLAR * params['scale_solar'].value
   IO2 = np.exp(-IO2 * params['scale_o2'].value )       
   ICIA = np.exp(-ICIA * (params['scale_o2'].value**2)/1E20 )  #np.exp(-ICIA* params['scale_cia'].value )         

   
   Isim = np.zeros(gridwidth+1)
   Isim = (( IO2 + ICIA ) *ISOLAR) + baseline
   residual =  Isim - Ioriginal
  
   #plt.plot(newobs_wavelength,Ioriginal,label='Iobs')
   #plt.plot(newobs_wavelength,Isim,label='Isim')
   #plt.legend()
   #plt.savefig('test_o2_fit.png')
   #plt.close()
   #exit() 

   return ( residual)
