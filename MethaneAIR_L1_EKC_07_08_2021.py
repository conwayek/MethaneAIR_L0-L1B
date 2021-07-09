# -*- coding: utf-8 -*-
import numpy as np
import datetime as dt
import struct
import os
import logging
from scipy.io import loadmat
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
#import time
from sys import exit
#SUPER GAUSSIAN ISRF
def supergaussian(x, w, k):     
    from math import gamma
    awk = k/( 2.0 * w * gamma(1/k)  )
    return awk * np.exp(-(abs(x/w) )**k)
     

#GAUSSIAN ISRF
def gaussian(x,width):
    return (1.0/(width*np.sqrt(2*np.pi) ) )  * np.exp(-0.5 * (x**2) / (width**2) )


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
    from scipy.interpolate import RegularGridInterpolator,interp1d
    from math import isclose
    from scipy import interpolate,optimize

    if(fitisrf == False):
        if isrf_lut0.shape != (len(isrf_w),len(isrf_dw0)):
            raise ValueError('isrf_lut dimension incompatible!')
            return np.full(w2.shape,np.nan)
        # make sure w1 and isrf_dw have the same resolution
        w1_step = np.median(np.diff(w1))
        isrf_dw_step = np.median(np.diff(isrf_dw0))

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
            isrf_dw_step = np.median(np.diff(isrf_dw0[iw,:]))
            
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
        isrf_dw_step = np.median(np.diff(isrf_dw0))
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
        isrf_dw_step = np.median(np.diff(isrf_dw0))
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

# below ReadSeq routines are from Jonathan Franklin
class Sequence():
    pass

class FrameMeta():
    pass

def ParseFrame(framenum,height,width,bindata):
#    print(framenum)
    Meta = FrameMeta()
    
    # Calc size of actual image minus meta data
    numpixels = (height-1)*width
    fmt = '<{}h'.format(numpixels)
    fmtsize = struct.calcsize(fmt)
    dataraw = struct.unpack_from(fmt,bindata[:fmtsize])
    data = np.reshape(dataraw,(height-1,width))
      
    # Grab time stamp
    temp = struct.unpack_from('<lh',bindata[-6:])
    Meta.timestamp = dt.datetime(1970,1,1) + \
              dt.timedelta(seconds=temp[-2],microseconds=temp[-1]*1000)
    # Grab Meta Data -- Not all of this seems to have real data in it.
    metaraw = bindata[fmtsize:-6]
    temp = struct.unpack_from('<{}h'.format(width),metaraw)
    metaraw = struct.pack('>{}h'.format(width),*temp)
      
    Meta.partNum = metaraw[2:34].decode('ascii').rstrip('\x00')
    Meta.serNum = metaraw[34:48].decode('ascii').rstrip('\x00')
    Meta.fpaType = metaraw[48:64].decode('ascii').rstrip('\x00')
#    print(Meta.partNum)
#    print(Meta.serNum)
#    print(Meta.fpaType)
    Meta.crc = struct.unpack_from('>I',metaraw[64:68])[0]
    Meta.frameCounter = struct.unpack_from('>i',metaraw[68:72])[0]
    Meta.frameTime = struct.unpack_from('>f',metaraw[72:76])[0]
    Meta.intTime = struct.unpack_from('>f',metaraw[76:80])[0]
    Meta.freq = struct.unpack_from('>f',metaraw[80:84])[0]
    Meta.boardTemp = struct.unpack_from('>f',metaraw[120:124])[0]
    Meta.rawNUC = struct.unpack_from('>H',metaraw[124:126])[0]
    Meta.colOff = struct.unpack_from('>h',metaraw[130:132])[0]
    Meta.numCols = struct.unpack_from('>h',metaraw[132:134])[0] + 1
    Meta.rowOff = struct.unpack_from('>h',metaraw[136:138])[0]
    Meta.numRows = struct.unpack_from('>h',metaraw[138:140])[0] + 1
    timelist = struct.unpack_from('>7h',metaraw[192:206])
    Meta.yr = timelist[0]
    Meta.dy = timelist[1]
    Meta.hr = timelist[2]
    Meta.mn = timelist[3]
    Meta.sc = timelist[4]
    Meta.ms = timelist[5]
    Meta.microsec = timelist[6]
    Meta.fpaTemp = struct.unpack_from('>f',metaraw[476:480])[0]
    Meta.intTimeTicks = struct.unpack_from('>I',metaraw[142:146])[0]
    
    return [data,Meta]

def ReadSeq(seqfile):
    ## Pull camera (ch4 v o2) and sequence timestamp from filename.
    temp = seqfile.split('/')[-1]
    temp = temp.split('_camera_')
    Seq = Sequence()
    Seq.Camera = temp[0]
    Seq.SeqTime = dt.datetime.strptime(temp[1].strip('.seq'),'%Y_%m_%d_%H_%M_%S')
    
    ## Open file for binary read
    fin = open(seqfile,'rb')
    binhead = fin.read(8192)
    
    ## Grab meta data for sequence file.
    temp = struct.unpack('<9I',binhead[548:548+36])
    Seq.ImageWidth = temp[0]
    Seq.ImageHeight = temp[1]
    Seq.ImageBitDepth = temp[2]
    Seq.ImageBitDepthTrue = temp[3]
    Seq.ImageSizeBytes = temp[4]
    Seq.NumFrames = temp[6]
    Seq.TrueImageSize = temp[8]
    Seq.NumPixels = Seq.ImageWidth*Seq.ImageHeight
    
    ## Read raw frames
    rawframes = [fin.read(Seq.TrueImageSize) for i in range(Seq.NumFrames)]
    fin.close()
    
    ## Process each frame -- dropping filler bytes before passing raw
    # print('Reading {} frames'.format(Seq.NumFrames))
    frames = [ParseFrame(i,Seq.ImageHeight,Seq.ImageWidth,
                         rawframes[i][:Seq.ImageSizeBytes+6]) \
                    for i in range(Seq.NumFrames)]
    
    Data = np.array([dd[0] for dd in frames])
    Meta = [dd[1] for dd in frames]
    
    return(Data,Meta,Seq)
# above ReadSeq routines are from Jonathan Franklin
    
class Dark():
    """
    place holder for an object for dark frame
    """
    pass

class Granule():
    """
    place holder for an object for calibrated level 1b granule
    """
    pass

class MethaneAIR_L1(object):
    """
    level 0 to level 1b processor for MethaneAIR
    """    
    
    def __init__(self,whichBand,l0DataDir,l1DataDir,
                 badPixelMapPath,radCalPath,wavCalPath,windowTransmissionPath,
                 solarPath,pixellimit,pixellimitX,spectraReffile,refitwavelength,
                 calibratewavelength,ISRFtype,fitisrf,calidata,fitSZA,SolZA,ALT,
                 xtol,ftol,xtrackaggfactor,atrackaggfactor):
        import math
        """
        whichBand:
            CH4 or O2
        l0DataDir:
            directory of level 0 data
        l1DataDir:
            directory of level 1 data
        badPixelMapPath:
            path to the bad pixel map
        radCalPath:
            path to the radiometric calibration coefficients
        wavCalPath:
            path to wavelength calibration coefficients
        windowTransmissionPath:
            path to the window transmission
        """
        #EAMON
     #   self.start_time = time.time()
        #READ IN PIXEL LIMITS FOR WAVELENGTH CALIBRATION, AND SOLAR REFERENCE FILE.
        self.solarRefFile = solarPath
        self.pixlimit = pixellimit
        self.pixlimitX = pixellimitX
        self.calibratewavelength  = calibratewavelength 
        self.refitwavelength = refitwavelength
        self.ISRF = ISRFtype
        self.spectraReffile=spectraReffile
        self.fitisrf = fitisrf
        self.calidata = calidata 
        self.fitSZA = fitSZA
        self.SZA = SolZA
        self.ALT = ALT
        self.badPixelMapPath = badPixelMapPath

        self.xtol=float(xtol)
        self.ftol=float(ftol)
        self.xtrackaggfactor = int(xtrackaggfactor)
        self.atrackaggfactor = int(atrackaggfactor)  
        
        self.whichBand = whichBand
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of level0->level1 for MethaneAIR '
                         +whichBand+ ' band')
        self.l0DataDir = l0DataDir
        if not os.path.isdir(l1DataDir):
            self.logger.warning(l1DataDir,' does not exist, creating one')
            os.mkdir(l1DataDir)
        self.l1DataDir = l1DataDir
        
        self.ncol = 1024
        self.nrow = 1280
        
        # bad pixel map
        self.logger.info('loading bad pixel map')
        self.badPixelMap = np.genfromtxt(badPixelMapPath,delimiter=',',dtype=np.int)
        self.logger.info('loading radiometric calibration coefficients')
        d = loadmat(radCalPath)['coef']
        
        # pad zero intercept and flip poly coeff order to be compatible with polyval
        radCalCoef = np.concatenate((np.zeros((*d.shape[0:2], 1)),d),axis=2)
        self.radCalCoef = np.flip(radCalCoef,axis=2)
        
        # wavelength calibration
        self.logger.info('loading wavelength calibration coefficients')
        spec_cal_fid = Dataset(wavCalPath)
        
        #EAMON
        self.isrf_dw0 = spec_cal_fid['delta_wavelength'][:].data
        self.isrf_lut = spec_cal_fid['isrf'][:,:,:].data
        self.isrf_w = spec_cal_fid['central_wavelength'][:].data
        

        if(self.whichBand == 'O2'):
            self.wavCalCoef = spec_cal_fid['pix2nm_polynomial'][:]
            
        else:
            self.wavCalCoef = spec_cal_fid['pix2nm_polynomial'][:]
            
            
            

        if self.wavCalCoef.shape[0] < self.wavCalCoef.shape[1]:
            self.logger.warning('you appear to be using an old version wavcal! tranposing...')
            self.wavCalCoef = self.wavCalCoef.T
        
        # wavelength calibration is independent of granules for now
        wavelength = np.array([np.polyval(self.wavCalCoef[i,::-1],np.arange(1,self.ncol+1))\
                           for i in range(self.nrow)])
        d = loadmat(windowTransmissionPath)
        # interpolate window transmission to detector wavelength
        f = interp1d(d['wavelength_nm'].squeeze(),d['transmission'].squeeze(),
                     fill_value='extrapolate')
        windowTransmission = f(wavelength)
        wavelength = np.ma.masked_array(wavelength,np.isnan(wavelength))
        windowTransmission = np.ma.masked_array(windowTransmission,
                                                np.isnan(windowTransmission))
        self.wavelength = wavelength
        self.windowTransmission = windowTransmission
        
        
                
    def F_stray_light_input(self,strayLightKernelPath,rowExtent,colExtent,
                            rowCenterMask=5,colCenterMask=7,nDeconvIter=1):
        """
        options for stray light correction
        strayLightKernelPath:
            path to the stray light deconvolution kernel
        rowExtent:
            max row difference away from center (0)
        colExtent:
            max column difference away from center (0)
        rowCenterMask:
            +/-rowCenterMask rows of the kernel will be masked to be 0
        colCenterMask:
            +/-colCenterMask columns of the kernel will be masked to be 0
        nDeconvIter:
            number of iterations in the Van Crittert deconvolution
        """
        d = loadmat(strayLightKernelPath)
        rowFilter = (d['rowAllGrid'].squeeze() >= -np.abs(rowExtent)) & \
        (d['rowAllGrid'].squeeze() <= np.abs(rowExtent))
        colFilter = (d['colAllGrid'].squeeze() >= -np.abs(colExtent)) & \
        (d['colAllGrid'].squeeze() <= np.abs(colExtent))
        reducedRow = d['rowAllGrid'].squeeze()[rowFilter]
        reducedCol = d['colAllGrid'].squeeze()[colFilter]
        strayLightKernel = d['medianStrayLight'][np.ix_(rowFilter,colFilter)]
        strayLightKernel[strayLightKernel<0] = 0.
        strayLightKernel = strayLightKernel/np.nansum(strayLightKernel)
        centerRowFilter = (reducedRow >= -np.abs(rowCenterMask)) & \
        (reducedRow <= np.abs(rowCenterMask))
        centerColFilter = (reducedCol >= -np.abs(colCenterMask)) & \
        (reducedCol <= np.abs(colCenterMask))
        strayLightKernel[np.ix_(centerRowFilter,centerColFilter)] = 0
        strayLightFraction = np.nansum(strayLightKernel)
        
        self.strayLightKernel = strayLightKernel
        self.strayLightFraction = strayLightFraction
        self.nDeconvIter = nDeconvIter
        
    def F_grab_dark(self,darkFramePath):
        """
        function to load dark measurement level 0 file
        darkFramePath:
            path to the file
        output is a Dark object
        """
        self.logger.info('loading dark seq file '+darkFramePath)
        (Data,Meta,Seq) = ReadSeq(darkFramePath)
        Data = np.transpose(Data,(2,1,0))
#        badPixelMap3D = np.repeat(self.badPixelMap[...,np.newaxis],Seq.NumFrames,axis=2)
#        darkData = np.nanmean(np.ma.masked_where(badPixelMap3D!=0,Data),axis=2)
#        darkStd = np.nanstd(np.ma.masked_where(badPixelMap3D!=0,Data),axis=2)
        darkData = np.nanmean(Data,axis=2)
        darkStd = np.nanstd(Data,axis=2)
#        darkData[self.badPixelMap!=0] = np.nan
#        darkStd[self.badPixelMap!=0] = np.nan

        """
        HERE NEW BAD PIXELS ARE DETERMINED VIA ANALYZING THE DARK CURRENT.
        BAD PIXEL MASKING IS APPLIED LATER ON IN THE GRANULE_PROCESSOR
        """
        
        #GET THE MEAN VALUE FOR ALL THE FRAMES AFTER MASKING WITH ALREADY KNOWN BAD PIXELS
        #DETERMINE MEDIAN CURRENT VALUES
        #ANY XTRACK/SPECTRAL INDEX WITH VALUE > 20% OF THIS IS NEW BAD PIXEL.
        #PRINT THIS OUT (TO FILE)?
    
        
        rowmean = np.nanmean(darkData,axis=1)
        self.newbadpixel = np.zeros((self.nrow,self.ncol))
        
        self.pixelflag=True 
        if(self.pixelflag == True):
            for i in range(self.pixlimitX[0],self.pixlimitX[1]):
                for j in range(self.ncol):
                    
                    if(darkData[i,j] > (1.80*rowmean[i])):
                        self.newbadpixel[i,j] = 2
                        self.badPixelMap[i,j] = 2
                        darkData[i,j] = np.nan

        dark = Dark()
        dark.data = darkData
        dark.noise = darkStd
        dark.nFrame = Seq.NumFrames
        dark.frameTime = np.array([Meta[i].frameTime for i in range(Seq.NumFrames)])
        dark.seqDateTime = Seq.SeqTime
        dark.frameDateTime = np.array([Meta[i].timestamp for i in range(Seq.NumFrames)])
        return dark
        
#    def F_error_input(self,darkOffsetDN=1500,ePerDN=4.6fitsp,readOutError=40):
#        """
#        try realistic error estimation parameters
#        removed given Jenna Samra's email on 2020/6/9 10:47 pm
#        """
#        self.darkOffsetDN = darkOffsetDN
#        self.ePerDN = ePerDN
#        self.readOutError = readOutError
        
    def F_granule_processor(self,granulePath,dark,ePerDN=4.6,timeOnly=False):
        from scipy.interpolate import interp1d
        from scipy import interpolate,optimize
        import wavecal_routines 
        """
        infile = '/Users/conwayek/Desktop/METHSAT_L0_L1B/L1B/MethaneAIR_L1B_CH4_20191112T212048_20191112T212049_20200710T165227.mat'
            #AGGREGATE THE OBSERVATION
        data = loadmat(infile)
        Data = data['radiance']
        """
        """
        working horse
        granulePath:
            path to a granule of level 0 data
        dark:
            a Dark object. Most useful part is dark.data, a nrow by ncol matrix 
            of dark frame for subtraction
        ePerDN:
            electrons per DN number
        timeOnly:
            if only save time stamps, no worries about other data
        output is a Granule object
        """
        #self.processing_start_time = time.time()

        if timeOnly:
            self.logger.info('loading seq file '+granulePath)
            (Data,Meta,Seq) = ReadSeq(granulePath)
            granuleFrameTime = np.array([Meta[i].frameTime for i in range(Seq.NumFrames)])
            granule = Granule()
            granule.nFrame = Seq.NumFrames
            granule.frameTime = granuleFrameTime
            granule.seqDateTime = Seq.SeqTime
            granule.frameDateTime = np.array([Meta[i].timestamp for i in range(Seq.NumFrames)])
            return granule
        darkData = dark.data
        darkStd = dark.noise
#        darkOffsetDN = self.darkOffsetDN
#        ePerDN = self.ePerDN
#        readOutError = self.readOutError
        self.logger.info('loading seq file '+granulePath)
        (Data,Meta,Seq) = ReadSeq(granulePath)
        Data = np.float32(np.transpose(Data,(2,1,0)))
        # estimate noise
        self.logger.info('estimate noise')
        Noise = np.sqrt((Data-darkData[...,np.newaxis])*ePerDN+(darkStd[...,np.newaxis]*ePerDN)**2)/ePerDN
#        Noise = np.sqrt((Data-darkOffsetDN)*ePerDN+readOutError**2)/ePerDN
        # remove dark
        Data = Data-darkData[...,np.newaxis]
        # mask bad pixels
        self.logger.info('mask bad pixels')
        badPixelMap3D = np.repeat(self.badPixelMap[...,np.newaxis],Seq.NumFrames,axis=2)
        Data[badPixelMap3D!=0] = np.nan
        Noise[badPixelMap3D!=0] = np.nan
    
        

#        Data = np.ma.masked_array(Data,
#                                  ((badPixelMap3D!=0) | (np.isnan(Data))) )
#        Noise = np.ma.masked_array(Noise,
#                                  ((badPixelMap3D!=0) | (np.isnan(Data))) )
        # normalize frame time, DN/s
        granuleFrameTime = np.array([Meta[i].frameTime for i in range(Seq.NumFrames)])
        granuleframeDateTime = np.array([Meta[i].timestamp for i in range(Seq.NumFrames)])
        Data = Data/granuleFrameTime[np.newaxis,np.newaxis,:]
        Noise = Noise/granuleFrameTime[np.newaxis,np.newaxis,:]
        
        # rad cal
        #self.logger.info('radiometric calibration')

        for i in range(self.nrow):
            for j in range(self.ncol):
                if np.isnan(self.radCalCoef[i,j,0]):
                    Data[i,j,:] = np.nan
                    Noise[i,j,:] = np.nan
                    continue
                Data[i,j,:] = np.polyval(self.radCalCoef[i,j,:],Data[i,j,:])
                Noise[i,j,:] = np.polyval(self.radCalCoef[i,j,:],Noise[i,j,:])
                if(Data[i,j,0] < -1e10):
                    Data[i,j,:] = np.nan
                if(Data[i,j,0] > 1e14):
                    Data[i,j,:] = np.nan


        
        if hasattr(self,'strayLightKernel'):
            self.logger.info('proceed with stray light correction')
            nDeconvIter = self.nDeconvIter
            strayLightKernel = self.strayLightKernel
            strayLightFraction = self.strayLightFraction
            from astropy.convolution import convolve_fft
            for iframe in range(Seq.NumFrames):
                tmpData = Data[:,:,iframe]
                for iDeconvIter in range(nDeconvIter):
                    tmpData = (Data[:,:,iframe]-convolve_fft(tmpData,strayLightKernel,normalize_kernel=False))\
                    /(1-strayLightFraction)
                Data[:,:,iframe] = tmpData
        else:
            self.logger.info('no data for stray light correction')
        # flip column order
        Data = Data[:,::-1,:]
        Noise = Noise[:,::-1,:]



        # window transmission correction
        for i in range(self.nrow):
            for j in range(self.ncol):
                if np.isnan(Data[i,j,0]):
                    Data[i,j,:] = np.nan
                else:
                    Data[i,j,:] = Data[i,j,:]/self.windowTransmission[i,j]
        if(self.calibratewavelength  == True and self.refitwavelength == False):
            self.logger.info('STARTING WAVELENGTH CALIBRATION')

            self.logger.info('CALIBRATING OBSERVED WAVELENGTH USING PRE-EXISTING FILE')
            row,shift,agg_x = [],[],[]
            for line in open(self.l1FitPath, 'r'):
                    values = [float(s) for s in line.split()]
                    row.append(values[0])
                    agg_x.append(values[1])
                    shift.append(values[2])  
                    
            row=np.array(row)
            agg_x=np.array(agg_x)
            row=row*agg_x[0]
            shift=np.array(shift)
                
            fit = interpolate.interp1d(row,shift)
                
            self.wvlcalibrated = np.zeros((Data.shape[0],Data.shape[2]),dtype=np.int8)
            self.wvlshiftdata = np.zeros((Data.shape[0],Data.shape[2]),dtype=np.float)

            for i in range(Data.shape[0]):
                if((i>=row[0]) and (i<=row[-1]) ):
                    spec_shift = fit(i)
                    self.wvlshiftdata[i,:] =  spec_shift
                    self.wvlcalibrated[i,:] = 1
                        


         
        # ELSE RECALIBRATE FOR NEW GRANULES
        elif(self.calibratewavelength == True and self.refitwavelength == True):
        
            if(self.whichBand == 'CH4'): 

                headerStr='MethaneAIR_L1B_CH4_'
                self.l1FitPath = os.path.join(self.l1DataDir,headerStr
                                      +np.min(granuleframeDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                      +np.max(granuleframeDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                      +str('wavecal_CH4.txt')) 

            else:

                headerStr='MethaneAIR_L1B_O2_'
                self.l1FitPath = os.path.join(self.l1DataDir,headerStr
                                      +np.min(granuleframeDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                      +np.max(granuleframeDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                      +str('wavecal_O2.txt')) 

            self.logger.info('RECALIBRATING OBSERVED WAVELENGTH: BEGIN FITTING')

            self.o2spectraReffile = 'o2_lut_1200-1750nm_0p02fwhm_1e21vcd_mesat.nc' 
            self.h2ospectraReffile = 'h2o_lut_1200-1750nm_0p02fwhm_1e21vcd_mesat.nc' 
            self.co2spectraReffile = 'co2_lut_1200-1750nm_0p02fwhm_1e21vcd_mesat.nc' 
            self.ch4spectraReffile = 'ch4_lut_1200-1750nm_0p02fwhm_1e17vcd_mesat.nc' 

            wavecal_routines.fitspectra(Data,self.l1FitPath,self.whichBand,self.solarRefFile,self.calidata,self.o2spectraReffile,self.spectraReffile,
                                        self.co2spectraReffile,self.h2ospectraReffile,self.ch4spectraReffile,
                                        self.fitSZA,self.SZA,self.ALT,self.l1DataDir,granuleframeDateTime,self.pixlimitX,
                                        self.isrf_lut,self.isrf_w,self.isrf_dw0,self.wavelength,self.pixlimit,self.fitisrf,
                                        self.ISRF,self.xtol,self.ftol,self.xtrackaggfactor,self.atrackaggfactor) 


            row,shift,agg_x = [],[],[]
            for line in open(self.l1FitPath, 'r'):
                    values = [float(s) for s in line.split()]
                    row.append(values[0])
                    agg_x.append(values[1])
                    shift.append(values[2])  
                    
            row=np.array(row)
            agg_x=np.array(agg_x)
            row=row*agg_x[0]
            shift=np.array(shift)
                
            fit = interpolate.interp1d(row,shift)
                
            self.wvlcalibrated = np.zeros((Data.shape[0],Data.shape[2]),dtype=np.int8)
            self.wvlshiftdata = np.zeros((Data.shape[0],Data.shape[2]),dtype=np.float)
            for i in range(Data.shape[0]):
                if((i>=row[0]) and (i<=row[-1]) ):
                    spec_shift = fit(i)
                    self.wvlshiftdata[i,:] =  spec_shift
                    self.wvlcalibrated[i,:] = 1
                    self.wavelength[i,:] = self.wavelength[i,:] + spec_shift       
        else:
            pass
        
        
        granule = Granule()
        granule.data = Data
        granule.noise = Noise
        granule.nFrame = Seq.NumFrames
        granule.frameTime = granuleFrameTime
        granule.seqDateTime = Seq.SeqTime
        granule.frameDateTime = np.array([Meta[i].timestamp for i in range(Seq.NumFrames)])
        return granule
    
    def F_block_reduce_granule(self,granule,acrossTrackAggregation=6,
                               alongTrackAggregation=10,ifKeepTail=True):
        """
        block-reduce gradule by aggregating in across/along track or row/frame dimensions
        graule:
            outputs from F_granule_processor, a Granule object
        across/alongTrackAggregation:
            as indicated by name
        ifKeepTail:
            whether to aggregate the leftover frames after block reduce
        """
        if acrossTrackAggregation == 1 and alongTrackAggregation == 1:
            self.logger.warning('No aggregation needs to be done. Why are you calling this function?')
            return granule
        from astropy.nddata.utils import block_reduce
        newGranule = Granule()
        newGranule.seqDateTime = granule.seqDateTime
        nFootprint = np.floor(granule.data.shape[0]/acrossTrackAggregation).astype(np.int)
        nTailRow = (granule.data.shape[0]-nFootprint*acrossTrackAggregation).astype(np.int)
        self.logger.info('%d'%granule.data.shape[0]+' rows will be reduced to %d'%nFootprint+' footprints')
        self.logger.info('The last %d'%nTailRow+' rows will be thrown away')
        nFrameAggregated = np.floor(granule.data.shape[2]/alongTrackAggregation).astype(np.int)
        nTailFrame = (granule.data.shape[2]-nFrameAggregated*alongTrackAggregation).astype(np.int)
        self.logger.info('%d'%granule.data.shape[2]+' frames in the granule will be reduced to %d'%nFrameAggregated+' aggregated frames')
        
        if not ifKeepTail or nTailFrame == 0:
            self.logger.info('The last %d'%nTailFrame+' frames will be thrown away')
            newGranule.data = block_reduce(granule.data,
                                           (acrossTrackAggregation,1,alongTrackAggregation),
                                           func=np.nanmean)
            sumSquare = block_reduce(np.power(granule.noise,2),
                                    (acrossTrackAggregation,1,alongTrackAggregation),
                                    func=np.nansum)
            newGranule.noise = np.sqrt(sumSquare/acrossTrackAggregation/alongTrackAggregation)
            newGranule.frameDateTime = block_reduce(granule.frameDateTime,
                                                    alongTrackAggregation,
                                                    func=np.nanmax)
            newGranule.frameTime = block_reduce(granule.frameTime,
                                                alongTrackAggregation,
                                                func=np.nansum)
            newGranule.nFrame = nFrameAggregated
        else:
            self.logger.info('The last %d'%nTailFrame+' frame will be aggregated')
            data1 = block_reduce(granule.data,
                                (acrossTrackAggregation,1,alongTrackAggregation),
                                func=np.nanmean)
            data2 = block_reduce(granule.data[...,-nTailFrame:],
                                 (acrossTrackAggregation,1,nTailFrame),
                                 func=np.nanmean)
            self.logger.debug('data1 shape is %d'%data1.shape[0]+', %d'%data1.shape[1]+', %d'%data1.shape[2])
            self.logger.debug('data2 shape is %d'%data2.shape[0]+', %d'%data2.shape[1]+', %d'%data2.shape[2])
            newGranule.data = np.concatenate((data1,data2),axis=2)
            sumSquare = block_reduce(np.power(granule.noise,2),
                                     (acrossTrackAggregation,1,alongTrackAggregation),
                                     func=np.nansum)
            noise1 = np.sqrt(sumSquare/acrossTrackAggregation/alongTrackAggregation)
            sumSquare = block_reduce(np.power(granule.noise[...,-nTailFrame:],2),
                                     (acrossTrackAggregation,1,nTailFrame),
                                     func=np.nansum)
            noise2 = np.sqrt(sumSquare/acrossTrackAggregation/nTailFrame)
            newGranule.noise = np.concatenate((noise1,noise2),axis=2)
            frameDateTime1 = block_reduce(granule.frameDateTime,
                                          alongTrackAggregation,
                                          func=np.nanmax)
            frameDateTime2 = block_reduce(granule.frameDateTime[-nTailFrame:],
                                          nTailFrame,
                                          func=np.nanmax)
            newGranule.frameDateTime = np.concatenate((frameDateTime1,frameDateTime2))
            frameTime1 = block_reduce(granule.frameTime,
                                      alongTrackAggregation,
                                      func=np.nansum)
            frameTime2 = block_reduce(granule.frameTime[-nTailFrame:],
                                      nTailFrame,
                                      func=np.nansum)
            newGranule.frameTime = np.concatenate((frameTime1,frameTime2))
            newGranule.nFrame = nFrameAggregated+1
            if len(newGranule.frameTime) != newGranule.nFrame:
                self.logger.error('this should not happen!')
        return newGranule
    
    def F_cut_granule(self,granule,granuleSeconds=10):
        """
        cut a granule into a list of granules with shorter, regular-time intervals
        graule:
            outputs from F_granule_processor, a Granule object
        granuleSeconds:
            length of cut granule in s
        """
        if hasattr(granule,'data'):
            data = granule.data
            noise = granule.noise
        #nFrame = granule.nFrame
        frameTime = granule.frameTime
        seqDateTime = granule.seqDateTime
        frameDateTime = granule.frameDateTime
        minDateTime = np.min(frameDateTime)
        maxDateTime = np.max(frameDateTime)
        minSecond = (minDateTime-minDateTime.replace(hour=0,minute=0,second=0,microsecond=0)).total_seconds()
        startSecond = np.floor(minSecond/granuleSeconds)*granuleSeconds
        minSecond = round((minSecond%granuleSeconds),7)
        startDateTime = minDateTime.replace(hour=0,minute=0,second=0,microsecond=0)+dt.timedelta(seconds=startSecond)
        #nGranule = np.floor((maxDateTime-startDateTime).total_seconds()/granuleSeconds)+1

        maxSecond = (maxDateTime-minDateTime).total_seconds()
        if((maxSecond%granuleSeconds)!=np.float(0)):
            nGranule = np.floor(maxSecond/granuleSeconds)
            extratime = np.float(maxSecond%granuleSeconds)
        else:
            nGranule = np.floor(maxSecond/granuleSeconds)
            extratime = np.float(0.0) 
        #endSecond = np.floor(maxSecond/granuleSeconds)*granuleSeconds
        #maxSecond = round((endSecond%granuleSeconds),7)

        nGranule = np.int16(nGranule)
        secondList = np.arange(nGranule+1)*np.float(granuleSeconds)
        #print('secondlist = ',secondList)
        granuleEdgeDateTimeList = np.array([startDateTime+dt.timedelta(seconds=secondList[i])+dt.timedelta(seconds=minSecond) for i in range(nGranule+1)])
        granuleEdgeDateTimeList[-1] = granuleEdgeDateTimeList[-1] + dt.timedelta(seconds=extratime)
        
        self.logger.info('cutting granule into %d'%nGranule+' shorter granules with length %d'%granuleSeconds +' seconds')
        granuleList = np.ndarray(shape=(nGranule),dtype=np.object_)
        for i in range(nGranule):
            if(i<(nGranule-1)):
                g0 = Granule()
                g0.seqDateTime = seqDateTime
                f = (frameDateTime >= granuleEdgeDateTimeList[i]) &\
                (frameDateTime < granuleEdgeDateTimeList[i+1])
                g0.nFrame = np.int16(np.sum(f))
                g0.frameDateTime = frameDateTime[f]
                g0.frameTime = frameTime[f]
                if hasattr(granule,'data'):
                    g0.data = data[...,f]
                    g0.noise = noise[...,f]
                granuleList[i] = g0
            elif(i==(nGranule-1)):
                g0 = Granule()
                g0.seqDateTime = seqDateTime
                f = (frameDateTime >= granuleEdgeDateTimeList[i]) &\
                (frameDateTime <= granuleEdgeDateTimeList[i+1])
                g0.nFrame = np.int16(np.sum(f))
                g0.frameDateTime = frameDateTime[f]
                g0.frameTime = frameTime[f]
                if hasattr(granule,'data'):
                    g0.data = data[...,f]
                    g0.noise = noise[...,f]
                granuleList[i] = g0
        return granuleList
    
    def F_save_L1B_time_only(self,granule,headerStr='MethaneAIR_L1B_CH4_timeonly_'):
        """
        save only the time stamps for calibrated data 
        granule:
            a Granule object generated by F_granule_processor or F_cut_granule
        headerStr:
            string that is different from the actual l1b files with real data 
        """
        from scipy.io import savemat
        l1FilePath = os.path.join(self.l1DataDir,headerStr
                                  +np.min(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +np.max(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
                                  +'.mat')
        GEOS_5_tau = np.array([(granule.frameDateTime[i]-dt.datetime(1985,1,1,0,0,0)).total_seconds()/3600.
                               for i in range(granule.nFrame)])
        granuleYear = np.array([granule.frameDateTime[i].year for i in range(granule.nFrame)])
        granuleMonth = np.array([granule.frameDateTime[i].month for i in range(granule.nFrame)])
        granuleDay = np.array([granule.frameDateTime[i].day for i in range(granule.nFrame)])
        granuleHour = np.array([granule.frameDateTime[i].hour for i in range(granule.nFrame)])
        granuleMinute = np.array([granule.frameDateTime[i].minute for i in range(granule.nFrame)])
        granuleSecond = np.array([granule.frameDateTime[i].second for i in range(granule.nFrame)])
        granuleMicrosecond = np.array([granule.frameDateTime[i].microsecond for i in range(granule.nFrame)])
        self.logger.info('saving time only .mat L1B file '+l1FilePath)
        savemat(l1FilePath,{'GEOS_5_tau':GEOS_5_tau,
                            'granuleYear':granuleYear,
                            'granuleMonth':granuleMonth,
                            'granuleDay':granuleDay,
                            'granuleHour':granuleHour,
                            'granuleMinute':granuleMinute,
                            'granuleSecond':granuleSecond,
                            'granuleMicrosecond':granuleMicrosecond})
    
    def F_save_L1B_mat(self,timestamp,granule,headerStr='MethaneAIR_L1B_CH4_',radianceOnly=False):
        """
        save calibrated data to level 1b file in .mat format for quick view
        granule:
            a Granule object generated by F_granule_processor or F_cut_granule
        headerStr:
            'MethaneAIR_L1B_CH4_' or 'MethaneAIR_L1B_O2_' or 'MethaneAIR_L1B_' 
        """
        from scipy.io import savemat
        import netCDF4 as nc4
        cwd=os.getcwd()
        l1FilePath = os.path.join(cwd,headerStr+timestamp+'.nc')
        """
        l1FilePath = os.path.join(self.l1DataDir,headerStr
                                  +np.min(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +np.max(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
                                  +'.mat')
        """
        GEOS_5_tau = np.array([(granule.frameDateTime[i]-dt.datetime(1985,1,1,0,0,0)).total_seconds()/3600.
                               for i in range(granule.nFrame)])
        granuleYear = np.array([granule.frameDateTime[i].year for i in range(granule.nFrame)])
        granuleMonth = np.array([granule.frameDateTime[i].month for i in range(granule.nFrame)])
        granuleDay = np.array([granule.frameDateTime[i].day for i in range(granule.nFrame)])
        granuleHour = np.array([granule.frameDateTime[i].hour for i in range(granule.nFrame)])
        granuleMinute = np.array([granule.frameDateTime[i].minute for i in range(granule.nFrame)])
        granuleSecond = np.array([granule.frameDateTime[i].second for i in range(granule.nFrame)])
        granuleMicrosecond = np.array([granule.frameDateTime[i].microsecond for i in range(granule.nFrame)])
        wavelength = np.tile(self.wavelength[...,np.newaxis],granule.nFrame).transpose([0,2,1])

        data = granule.data.transpose([0,2,1])
        noise = granule.noise.transpose([0,2,1])
        # flip row order if O2 camera
        if self.whichBand == 'O2':
            wavelength = wavelength[::-1,...]
            data = data[::-1,...]
            noise = noise[::-1,...]
        self.logger.info('saving .mat L1B file '+l1FilePath)
        if radianceOnly:
            savemat(l1FilePath,{'radiance':np.asfortranarray(data).astype(np.float32),
                            'GEOS_5_tau':GEOS_5_tau,
                            'granuleYear':granuleYear,
                            'granuleMonth':granuleMonth,
                            'granuleDay':granuleDay,
                            'granuleHour':granuleHour,
                            'granuleMinute':granuleMinute,
                            'granuleSecond':granuleSecond,
                            'granuleMicrosecond':granuleMicrosecond})
            return

        """ 
        savemat(l1FilePath,{'wavelength':np.asfortranarray(wavelength),
                            'radiance':np.asfortranarray(data),
                            'radiance_error':np.asfortranarray(noise),
                            'GEOS_5_tau':GEOS_5_tau,
                            'granuleYear':granuleYear,
                            'granuleMonth':granuleMonth,
                            'granuleDay':granuleDay,
                            'granuleHour':granuleHour,
                            'granuleMinute':granuleMinute,
                            'granuleSecond':granuleSecond,
                            'granuleMicrosecond':granuleMicrosecond})
        """
        f = nc4.Dataset(l1FilePath,'w', format='NETCDF4')
        
        f.createDimension('ncol', self.ncol)
        f.createDimension('nrow', self.nrow)
        f.createDimension('nframe', granule.nFrame)
          
 
        wvl = f.createVariable('wavelength', 'f4', ( 'nrow', 'nframe', 'ncol'   )   )
        rad = f.createVariable('radiance', 'f4', ( 'nrow', 'nframe', 'ncol'   )   )
        raderr = f.createVariable('radiance_error', 'f4', ( 'nrow', 'nframe', 'ncol'   )   )
        geos = f.createVariable('GEOS_5_tau', 'f8', 'nframe'   )
        year = f.createVariable('granuleYear', 'i4', 'nframe'   )
        month = f.createVariable('granuleMonth', 'i4', 'nframe'   )
        day = f.createVariable('granuleDay', 'i4', 'nframe'   )
        hour = f.createVariable('granuleHour', 'i4', 'nframe'   )
        minute = f.createVariable('granuleMinute', 'i4', 'nframe'   )
        second = f.createVariable('granuleSecond', 'i4', 'nframe'   )
        micro = f.createVariable('granuleMicrosecond', 'i4', 'nframe'   )
        
        wvl[:,:,:] = wavelength
        rad[:,:,:] = data 
        raderr[:,:,:] = noise
        geos[:] = GEOS_5_tau
        year[:] = granuleYear
        month[:] = granuleMonth
        day[:] = granuleDay
        hour[:] = granuleHour
        minute[:] = granuleMinute
        second[:] = granuleSecond
        micro[:] = granuleMicrosecond
        f.close() 
        
    def F_save_L1B(self,granule,headerStr='MethaneAIR_L1B_CH4_'):
        """
        save calibrated data to level 1b file
        granule:
            a Granule object generated by F_granule_processor or F_cut_granule
        headerStr:
            'MethaneAIR_L1B_CH4_' or 'MethaneAIR_L1B_O2_' or 'MethaneAIR_L1B_' 
        """
        from pysplat import level1
        l1FilePath = os.path.join(self.l1DataDir,headerStr
                                  +np.min(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +np.max(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
                                  +'.nc')
        GEOS_5_tau = np.array([(granule.frameDateTime[i]-dt.datetime(1985,1,1,0,0,0)).total_seconds()/3600.
                               for i in range(granule.nFrame)])
        wavelength = np.tile(self.wavelength[...,np.newaxis],granule.nFrame).transpose([0,2,1])
        data = granule.data.transpose([0,2,1])
        noise = granule.noise.transpose([0,2,1])
        # flip row order if O2 camera
        if self.whichBand == 'O2':
            wavelength = wavelength[::-1,...]
            data = data[::-1,...]
            noise = noise[::-1,...]
        l1 = level1(l1FilePath,
                    lon=np.zeros((self.nrow,granule.nFrame)),
                    lat=np.zeros((self.nrow,granule.nFrame)),
                    obsalt=np.zeros(granule.nFrame),
                    time=GEOS_5_tau)
        l1.add_radiance_band(wvl=wavelength,
                             rad=data,
                             rad_err=noise)
        l1.close()
        
    def read_csv(self,infile,header=True):

        import pysplat
        import csv

        with open(infile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True
            output = {} ; keys = []
            for row in csv_reader:
                if(first):
                    if(header):
                        for name in row:
                            output[name] = [] ; keys.append(name) ; nvar = len(keys)
                    else:
                        nvar = len(row)
                        for n in range(nvar):
                            output['col_'+str(n)] = [] ; keys.append('col_'+str(n))
                        for n in range(nvar):
                            output[keys[n]].append(row[n])
                    first = False

                else:
                    for n in range(nvar):
                        output[keys[n]].append(row[n])
        return output
        

    def read_geolocation(self,geodir,timestamp,readviewgeo=True):

        band = self.whichBand
        filename=os.path.join(geodir,'MethaneAIR_L1B_'+band+'_'+timestamp+'.nc')
        data = Dataset(filename,'r')
        # need to flip the o2 xtrack ordering for viewing geometries and lon/lat data. 
        # the radiance is already flipped
        if( (self.whichBand == 'O2')  and (self.ortho_step == 'optimized')  ):
            zsurf = data['SurfaceAltitude'][::-1,:] * 1e-3
               
            lat = data['Latitude'][::-1,:]

            lon = data['Longitude'][::-1,:]

            lat_ll = data['CornerLatitude'][0,:,:]
            lat_ul = data['CornerLatitude'][2,:,:]
            lat_ur = data['CornerLatitude'][3,:,:]
            lat_lr = data['CornerLatitude'][1,:,:]

            lon_ll = data['CornerLongitude'][0,:,:]
            lon_ul = data['CornerLongitude'][2,:,:]
            lon_ur = data['CornerLongitude'][3,:,:]
            lon_lr = data['CornerLongitude'][1,:,:]

            clon = np.zeros(( lat.shape[0], lat.shape[1],4 ))
            clat = np.zeros(( lat.shape[0], lat.shape[1],4 ))
             
            clon[:,:,0] = lon_ll[::-1,:]
            clon[:,:,1] = lon_ul[::-1,:]
            clon[:,:,2] = lon_ur[::-1,:]
            clon[:,:,3] = lon_lr[::-1,:]

            clat[:,:,0] = lat_ll[::-1,:]
            clat[:,:,1] = lat_ul[::-1,:]
            clat[:,:,2] = lat_ur[::-1,:]
            clat[:,:,3] = lat_lr[::-1,:]

            lon_ll=None
            lon_ul=None
            lon_ur=None
            lon_lr=None
            lat_ll=None
            lat_ul=None
            lat_ur=None
            lat_lr=None


            ac_lon = data['Aircraft_Longitude'][:]
            ac_lat = data['Aircraft_Latitude'][:]
            ac_bore = data['Aircraft_PixelBore'][:,:,::-1]
            ac_altwgs84 = data['Aircraft_AltitudeAboveWGS84'][:] * 1e-3
            ac_altsurf = data['Aircraft_AltitudeAboveSurface'][:] * 1e-3
            ac_pos = data['Aircraft_Pos'][::-1,:]
            ac_surfalt = data['Aircraft_SurfaceAltitude'][:] * 1e-3


            if(readviewgeo):
                saa = data['SolarAzimuthAngle'][::-1,:]
                sza = data['SolarZenithAngle'][::-1,:]
                vaa = data['ViewingAzimuthAngle'][::-1,:]
                vza = data['ViewingZenithAngle'][::-1,:]
                aza = np.zeros((vza.shape))
                # Compute VLIDORT relative azimuth (180 from usual def)
                aza = vaa - (saa + 180.0)
                idv = np.isfinite(aza)
                if( (aza[idv] < 0.0).any() or (aza[idv] > 360.0).any() ):
                    aza[aza<0.0] = aza[aza<0.0]+360.0
                    aza[aza>360.0] = aza[aza>360.0]-360.0
                return clon,clat,lon,lat,zsurf,sza,vza,aza,saa,vaa,ac_lon,ac_lat,ac_bore,ac_altwgs84,ac_altsurf,ac_pos,ac_surfalt
            else:
                return clon,clat,lon,lat,zsurf
        else:
            zsurf = data['SurfaceAltitude'][:,:] * 1e-3
               
            lat = data['Latitude'][:,:]

            lon = data['Longitude'][:,:]

            lat_ll = data['CornerLatitude'][0,:,:]
            lat_ul = data['CornerLatitude'][2,:,:]
            lat_ur = data['CornerLatitude'][3,:,:]
            lat_lr = data['CornerLatitude'][1,:,:]

            lon_ll = data['CornerLongitude'][0,:,:]
            lon_ul = data['CornerLongitude'][2,:,:]
            lon_ur = data['CornerLongitude'][3,:,:]
            lon_lr = data['CornerLongitude'][1,:,:]

            clon = np.zeros(( lat.shape[0], lat.shape[1],4 ))
            clat = np.zeros(( lat.shape[0], lat.shape[1],4 ))
             
            clon[:,:,0] = lon_ll
            clon[:,:,1] = lon_ul
            clon[:,:,2] = lon_ur
            clon[:,:,3] = lon_lr

            clat[:,:,0] = lat_ll
            clat[:,:,1] = lat_ul
            clat[:,:,2] = lat_ur
            clat[:,:,3] = lat_lr

            lon_ll=None
            lon_ul=None
            lon_ur=None
            lon_lr=None
            lat_ll=None
            lat_ul=None
            lat_ur=None
            lat_lr=None

            ac_lon = data['Aircraft_Longitude'][:]
            ac_lat = data['Aircraft_Latitude'][:]
            ac_bore = data['Aircraft_PixelBore'][:,:,:]
            ac_altwgs84 = data['Aircraft_AltitudeAboveWGS84'][:] * 1e-3
            ac_altsurf = data['Aircraft_AltitudeAboveSurface'][:] * 1e-3
            ac_pos = data['Aircraft_Pos'][:,:]
            ac_surfalt = data['Aircraft_SurfaceAltitude'][:] * 1e-3


            if(readviewgeo):
                saa = data['SolarAzimuthAngle'][:,:]
                sza = data['SolarZenithAngle'][:,:]
                vaa = data['ViewingAzimuthAngle'][:,:]
                vza = data['ViewingZenithAngle'][:,:]
                aza = np.zeros((vza.shape))
                # Compute VLIDORT relative azimuth (180 from usual def)
                aza = vaa - (saa + 180.0)
                idv = np.isfinite(aza)
                if( (aza[idv] < 0.0).any() or (aza[idv] > 360.0).any() ):
                    aza[aza<0.0] = aza[aza<0.0]+360.0
                    aza[aza>360.0] = aza[aza>360.0]-360.0
                return clon,clat,lon,lat,zsurf,sza,vza,aza,saa,vaa,ac_lon,ac_lat,ac_bore,ac_altwgs84,ac_altsurf,ac_pos,ac_surfalt
            else:
                return clon,clat,lon,lat,zsurf


        """
        # File locations
        demfile = os.path.join(geodir,'DEM/MethaneAIR_L1B_'+band+'_'+timestamp+'.csv')
        latfile = os.path.join(geodir,'Lat/MethaneAIR_L1B_'+band+'_'+timestamp+'.csv')
        lonfile = os.path.join(geodir,'Lon/MethaneAIR_L1B_'+band+'_'+timestamp+'.csv')

        # Open DEM
        d = self.read_csv(demfile)
        xmx = len(d.keys())
        tmx = len(d['V1'])

        # Load DEM
        zsurf = np.zeros((xmx,tmx))
        for x in range(xmx):
            varname = 'V'+str(x+1)
            zsurf[x,:] = d[varname][:]
        zsurf = zsurf*1e-3

        # Load Latitude
        d = self.read_csv(latfile)
        lat = np.zeros((xmx,tmx))
        for x in range(xmx):
            varname = 'V'+str(x+1)
            lat[x,:] = d[varname][:]

        # Load Longitude
        d = self.read_csv(lonfile)
        lon = np.zeros((xmx,tmx))
        for x in range(xmx):
            varname = 'V'+str(x+1)
            lon[x,:] = d[varname][:]

        # Reading viewing geometry
        if(readviewgeo):
            
            # Path to viewing geometry files
            szafile = os.path.join(geodir,'SZA/MethaneAIR_L1B_'+band+'_'+timestamp+'.csv')
            vzafile = os.path.join(geodir,'VZA/MethaneAIR_L1B_'+band+'_'+timestamp+'.csv')
            saafile = os.path.join(geodir,'SAA/MethaneAIR_L1B_'+band+'_'+timestamp+'.csv')
            vaafile = os.path.join(geodir,'VAA/MethaneAIR_L1B_'+band+'_'+timestamp+'.csv')

            # Load SZA
            d = self.read_csv(szafile)
            sza = np.zeros((xmx,tmx))
            for x in range(xmx):
                varname = 'V'+str(x+1)
                sza[x,:] = d[varname][:]

            # Load VZA
            d = self.read_csv(vzafile)
            vza = np.zeros((xmx,tmx))
            for x in range(xmx):
                varname = 'V'+str(x+1)
                vza[x,:] = d[varname][:]

            # Load SAA
            d = self.read_csv(saafile)
            saa = np.zeros((xmx,tmx))
            for x in range(xmx):
                varname = 'V'+str(x+1)
                saa[x,:] = d[varname][:]
            
            # Load VAA
            d = self.read_csv(vaafile)
            vaa = np.zeros((xmx,tmx))
            for x in range(xmx):
                varname = 'V'+str(x+1)
                vaa[x,:] = d[varname][:]
            
            # Compute VLIDORT relative azimuth (180 from usual def)
            aza = vaa - (saa + 180.0)
            idv = np.isfinite(aza)
            if( (aza[idv] < 0.0).any() or (aza[idv] > 360.0).any() ):
                aza[aza<0.0] = aza[aza<0.0]+360.0
                aza[aza>360.0] = aza[aza>360.0]-360.0

            return lon,lat,zsurf,sza,vza,aza,saa,vaa
        else:
            return lon,lat,zsurf
        """


            
    def write_splat_l1_coadd(self,l1_rad_dir,l1_geodir,time_stamp,l1_outdir,xtrk_aggfac=1,atrk_aggfac=1,wvl_file=None,ortho_step='avionics'):

        ''' Create splat level1 file and corresponding ISRF LUT

            ARGS:
              l1_rad_dir: Directory with calibrated radiances
              l1_geodir: Directory with geolocation information
              l1_flight_file: File containing the flight observation altitude and sub-platform coordinates
              time_stamp: time stamp to process  
                           <START_TIME>_<END_TIME>_<PROCESSING_TIME> 
                           where time format is YYYYMMDDTHHMMSS
              l1_outdir: Output directory

            OPT ARGS:
              xtrk_aggfac: Cross-track aggregation factor
              atrk_aggfac: Along-track aggregation factor
              wvl_file: File with wavelength grid
              band (str): MethaneAIR Band ('CH4' (default) or 'O2')
        '''

        import pysplat
        import os
        from skimage.measure import block_reduce
        self.ortho_step = ortho_step 
        band = self.whichBand
        # Set files
        l1_matfile = os.path.join(l1_rad_dir,'MethaneAIR_L1B_'+str(band)+'_'+str(time_stamp)+'.nc')
        l1_outfile = os.path.join(l1_outdir,'MethaneAIR_L1B_'+str(band)+'_'+str(time_stamp)+'.nc')
        
        if(self.whichBand == 'CH4'):
            headerStr='MethaneAIR_L1B_CH4_'
        else:
            headerStr='MethaneAIR_L1B_O2_'

            
        
        # Load the matlab file
        l1data = Dataset(l1_matfile,'r')
        
        # Get Native dimensions
        #wmx = l1data['radiance'].shape[1]
        #xmx = l1data['radiance'].shape[0]
        #tmx = l1data['radiance'].shape[2]
        
        wmx = l1data['radiance'].shape[2]
        xmx = l1data['radiance'].shape[0]
        tmx = l1data['radiance'].shape[1]
        
        # Compute tau time from input data
        tau_native = np.zeros(tmx)
        for t in range(tmx):
            nymd = l1data['granuleYear'][t]*10000 \
                 + l1data['granuleMonth'][t]*100  \
                 + l1data['granuleDay'][t]
            tau_native[t] = pysplat.time.nymd2tau(nymd)[0]        \
                          + float(l1data['granuleHour'][t])        \
                          + float(l1data['granuleMinute'][t])/60.0 \
                          + float(l1data['granuleSecond'][t])/3600.0 \
                          + float(l1data['granuleMicrosecond'][t]/(3600e6))        
        # Block pads with zeros and therefore does not correctly normalize - do this here
        norm_3d = block_reduce(np.ones(l1data['radiance'].shape),        \
                               block_size=(xtrk_aggfac, atrk_aggfac, 1),\
                               func=np.mean                             ).transpose((0,2,1))
        
        # Valid pixels
        valpix = np.zeros(l1data['radiance'].shape)
        idv = np.logical_and(np.isfinite(l1data['radiance'][:,:,:]),l1data['radiance'][:,:,:]>0.0)
        valpix[idv] = 1.0
        valpix_agg = block_reduce(valpix,                       \
                                  block_size=(xtrk_aggfac, atrk_aggfac, 1),\
                                  func=np.mean                             ).transpose((0,2,1))
        valpix_agg = valpix_agg / norm_3d

        # Coadd radiance data
        rad = block_reduce(l1data['radiance'],                       \
                           block_size=(xtrk_aggfac, atrk_aggfac, 1),\
                           func=np.mean                             ).transpose((0,2,1))
        rad = rad / norm_3d
        # Only keep radiances where all 6x10 present
        rad[valpix_agg <0.99999999999999999999] = np.nan
        rad = rad[::-1,:,:]        


        # For now assume native SNR of ~100
        #rad_err = l1data['radiance_error']#rad/100.0/np.sqrt(xtrk_aggfac*atrk_aggfac)
        #rad_err = rad/100.0/np.sqrt(xtrk_aggfac*atrk_aggfac)
        rad_err =l1data['radiance_error'] 

        norm_3d = block_reduce(np.ones(l1data['radiance_error'].shape),        \
                               block_size=(xtrk_aggfac, atrk_aggfac, 1),\
                               func=np.mean                             ).transpose((0,2,1))

        rad_err = block_reduce(l1data['radiance_error'],                       \
                           block_size=(xtrk_aggfac, atrk_aggfac, 1),\
                           func=np.mean                             ).transpose((0,2,1))
        rad_err = rad_err / norm_3d
        rad_err[valpix_agg <0.99999999999999999999] = np.nan
        rad_err = rad_err / np.sqrt(xtrk_aggfac*atrk_aggfac) 

        rad_err = rad_err[::-1,:,:]        


        # Get output dimensions from radiance aggregation
        nx = rad.shape[0]
        nt = rad.shape[2]
        

        
        xmesh,ymesh = np.meshgrid(np.arange(nx),np.arange(rad.shape[2]),indexing='ij')

        # Load calibration data
        bad_pix_native = self.load_caldata()
        
        # Degrade the bad pixel flags,x,w
        bad_pix = block_reduce(bad_pix_native,block_size=(xtrk_aggfac,1),func=np.max)
        
        # Set the radiance flags
        rad_flags = np.zeros((nx,wmx,nt),np.int16)
        for t in range(nt):
            rad_flags[:,:,t] = bad_pix
        rad_flags = rad_flags.transpose((0,2,1))

        # Wavelength grid
        wvl = np.zeros((nx,nt,wmx))

        """
        # Check if we need to add a shift
        if(not wvl_file is None):

            # Load file
            d = Dataset(wvl_file,'r')
            wvl_1x1 = d.variables['wvl'][:] # (x,w)
            d.close()
            
            # Degrade file to cross track
            norm_2d = block_reduce(np.ones(wvl_1x1.shape),block_size=(xtrk_aggfac,1),func=np.mean)
            wvl_ccd  = block_reduce(wvl_1x1,block_size=(xtrk_aggfac,1),func=np.mean)
            wvl_ccd = wvl_ccd / norm_2d
            
            # Add shift
            for x in range(nx):
                for t in range(nt):
                    wvl[x,t,:] = wvl_ccd[x,:].squeeze()
            
        else:
        """

        wvl = l1data['wavelength']
        norm_3d = block_reduce(np.ones(wvl.shape),        \
                               block_size=(xtrk_aggfac, atrk_aggfac, 1),\
                               func=np.mean                             )#.transpose((0,2,1))
        wvl = block_reduce(wvl,                       \
                           block_size=(xtrk_aggfac, atrk_aggfac, 1),\
                           func=np.mean                             )#.transpose((0,2,1))
        wvl = wvl / norm_3d
        



        rad = np.transpose(rad,(0,2,1))
        rad_err = np.transpose(rad_err,(0,2,1))




        # Get pixel geolocation on aggregated grid
        corner_lon_native,corner_lat_native,lon_native,lat_native,zsurf_native,sza,vza,aza,saa,vaa,ac_lon,ac_lat,ac_bore,ac_altwgs84,ac_altsurf,ac_pos,ac_surfalt = self.read_geolocation(l1_geodir,time_stamp,readviewgeo=True)

        ####################3
        
        # Time
        norm_1d = block_reduce(np.ones(tau_native.shape),block_size=(atrk_aggfac,),func=np.mean)
        tau = block_reduce(tau_native,block_size=(atrk_aggfac,),func=np.mean)
        tau = tau / norm_1d

        ac_lon = block_reduce(ac_lon,block_size=(atrk_aggfac,),func=np.mean)
        ac_lon = ac_lon / norm_1d

        ac_lat = block_reduce(ac_lat,block_size=(atrk_aggfac,),func=np.mean)
        ac_lat = ac_lat / norm_1d

        ac_altwgs84 = block_reduce(ac_altwgs84,block_size=(atrk_aggfac,),func=np.mean)
        ac_altwgs84 = ac_altwgs84 / norm_1d

        ac_altsurf = block_reduce(ac_altsurf,block_size=(atrk_aggfac,),func=np.mean)
        ac_altsurf = ac_altsurf / norm_1d

        ac_surfalt = block_reduce(ac_surfalt,block_size=(atrk_aggfac,),func=np.mean)
        ac_surfalt = ac_surfalt / norm_1d

        ####################3

        norm_2d = block_reduce(np.ones(ac_pos.shape),block_size=(atrk_aggfac,1),func=np.mean)

        ac_pos = block_reduce(ac_pos,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; ac_pos = ac_pos/norm_2d

        ####################3
 
        norm_2d = block_reduce(np.ones(lon_native.shape),block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean)

        zsurf = block_reduce(zsurf_native,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; zsurf = zsurf/norm_2d

        lon = block_reduce(lon_native,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; lon = lon/norm_2d
        lat = block_reduce(lat_native,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; lat = lat/norm_2d


        sza = block_reduce(sza,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; sza = sza/norm_2d
        saa = block_reduce(saa,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; saa = saa/norm_2d
        vaa = block_reduce(vaa,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; vaa = vaa/norm_2d
        vza = block_reduce(vza,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; vza = vza/norm_2d
        aza = block_reduce(aza,block_size=(xtrk_aggfac,atrk_aggfac),func=np.mean) ; aza = aza/norm_2d

        ####################3

        norm_3d = block_reduce(np.ones(corner_lon_native.shape),  block_size=(xtrk_aggfac, atrk_aggfac, 1),func=np.mean)
        corner_lon = block_reduce(corner_lon_native,block_size=(xtrk_aggfac,atrk_aggfac,1),func=np.mean) ; corner_lon = corner_lon/norm_3d
        corner_lat = block_reduce(corner_lat_native,block_size=(xtrk_aggfac,atrk_aggfac,1),func=np.mean) ; corner_lat = corner_lat/norm_3d

        ####################3
        ac_bore = ac_bore.transpose((2,1,0))
        norm_3d = block_reduce(np.ones(ac_bore.shape),  block_size=(xtrk_aggfac, atrk_aggfac, 1),func=np.mean)
        ac_bore = block_reduce(ac_bore,block_size=(xtrk_aggfac,atrk_aggfac,1),func=np.mean) ; ac_bore = ac_bore/norm_3d

        # Get flight location
        #obslon,obslat,obsalt = self.get_flight_location(l1_flight_file,tau)
        # Create the level1 Object

        lon = lon.transpose(1,0)
        lat = lat.transpose(1,0)
        corner_lat = corner_lat.transpose(1,0,2)
        corner_lon = corner_lon.transpose(1,0,2)
        sza = sza.transpose(1,0)
        saa = saa.transpose(1,0)
        vaa = vaa.transpose(1,0)
        vza = vza.transpose(1,0)
        aza = aza.transpose(1,0)
        zsurf = zsurf.transpose(1,0)
        ac_pos = ac_pos.transpose(1,0)


        l1 = pysplat.level1(l1_outfile,lon,lat,ac_altwgs84,tau,ac_lon,ac_lat,ac_pos,ac_surfalt,ac_altsurf,ac_bore,optbenchT=None,clon=corner_lon,clat=corner_lat)
        # Set the surface altitude
        l1.set_2d_geofield('SurfaceAltitude', zsurf)
        l1.set_2d_geofield('SolarZenithAngle', sza)
        l1.set_2d_geofield('SolarAzimuthAngle', saa)
        l1.set_2d_geofield('ViewingZenithAngle', vza)
        l1.set_2d_geofield('ViewingAzimuthAngle', vaa)
        l1.set_2d_geofield('RelativeAzimuthAngle', aza)

        # Write the band
        l1.add_radiance_band(wvl,rad,rad_err=rad_err,rad_flag=rad_flags)

        # Close file
        l1.close()
        
        return rad
        
    
    def load_caldata(self,indir='caldata',band='CH4'):

        outp ={}
        #bad pixel map = self.badPixelMapPath 

        # Path to files
        badpix_file = self.badPixelMapPath #os.path.join(indir,band+'_bad_pix.csv')
        #gain_file = os.path.join(indir,band+'_gain.csv')
        #gain_err_file = os.path.join(indir,band+'_gain_err.csv')

        # Load Bad pixel map
        d = self.read_csv(badpix_file,header=False)
        wmx = len(d.keys())
        xmx = len(d['col_1'][:])

        # Load Bad Pixels
        bad_pix = np.zeros((xmx,wmx),dtype=np.int16)
        for w in range(wmx):
            fldname = 'col_'+str(w)
            ts = np.array(['nan' if x == 'NaN' else x for x in d[fldname]],dtype=np.int16)
            #print(w,ts.min(),ts.max())
            bad_pix[:,w] = ts
        """
        d = read_csv(gain_err_file,header=False)
        gain_err = np.zeros((xmx,wmx),dtype=np.float)
        for w in range(wmx):
            fldname = 'col_'+str(w)
            ts = np.array(['nan' if x == 'NaN' else x for x in d[fldname]],dtype=np.float)
            print(w,ts.min(),ts.max())
            gain_err[:,w] = ts
        """

        return bad_pix#, gain_err
    
        
    
        
    def get_flight_location(self,geofile,tau_in):

        import pysplat

        # Read CSV
        d=self.read_csv(geofile)

        # Compute time
        npt = len(d['time'])
        tau = np.zeros(npt)

        for n in range(npt):

            # Process string
            f = d['time'][n].split(' ')
            ymd = f[0].split('-')
            hms = f[1].split(':')
            tau[n] = pysplat.time.nymd2tau(int(ymd[0]+ymd[1]+ymd[2]))[0] \
                   +np.float64(hms[0])+np.float64(hms[1])/60.0+np.float64(hms[2])/3600.0
        tau = np.round(tau,8)
        # Convert N/A to nans
        fldname = ['geoid_height_WGS84', 'alt_geoid', 'lon', 'lat']
        output = {}
        #print(d.keys())
        for fld in fldname:
            ts = ['nan' if x == 'NA' else x for x in d[fld]] ; d[fld] = ts
            output[fld] = np.array(d[fld],dtype=np.float)

        # Get values
        obsalt = (output['geoid_height_WGS84']+output['alt_geoid'])*1e-3
        obslat = output['lat']
        obslon = output['lon']
        idv = np.logical_and.reduce([np.isfinite(obsalt),np.isfinite(obslat),np.isfinite(obslon)])

        # Interpolate to observation time
        f_alt = interp1d(tau[idv],obsalt[idv],bounds_error=False)
        f_lon = interp1d(tau[idv],obslon[idv],bounds_error=False)
        f_lat = interp1d(tau[idv],obslat[idv],bounds_error=False)
        
        #print(tau_in)
        #print(np.min(tau[idv]),np.max(tau[idv]))

        return f_lon(tau_in), f_lat(tau_in), f_alt(tau_in)        
        
        
        
    def fitspectra(self,Data,frameDateTime):
        
        from scipy import interpolate,optimize
        from lmfit import minimize, Parameters, Parameter, printfuncs, fit_report
        from scipy.interpolate import RegularGridInterpolator
        import math
        #AGGREGATE THE OBSERVATION
        #radiance = Data*1e-14
        #rad_obs = np.mean(radiance,axis=2)
        radiance = np.mean(Data,axis=2)
        rad_obs = radiance*1e-14
        

        #NUMBER OF FRAMES CAN BE USED TO GET FIRST GUESS ON THE VALUE OF SOLAR SCALING
        self.numframes = len(Data[0,0,:])
        
        ##############################
        #SOLAR DATA
        ncid = Dataset(self.solarRefFile, 'r')
        if(self.whichBand == 'CH4'):
            f = (ncid['Wavelength'][:] > 1590) & (ncid['Wavelength'][:] < 1700)
            self.refWavelength = ncid['Wavelength'][:][f].data
            self.refSolarIrradiance = ncid['SSI'][:][f].data
        else:
            f = (ncid['Wavelength'][:] > 1245) & (ncid['Wavelength'][:] < 1310)
            self.refWavelength = ncid['Wavelength'][:][f].data
            self.refSolarIrradiance = ncid['SSI'][:][f].data
        ncid.close()
        ##############################
        
        
        ##############################
        # CROSS SECTIONAL DATA
        """
        ncid = Dataset(self.spectraReffile, 'r')
        specwave        = ncid.variables['Wavelength'][:]
        if(self.whichBand == 'O2'):
            O2  = ncid.variables['O2'][:]
            #CIA  = ncid.variables['CIAO2AIR'][:]
            #CIA  = ncid.variables['CIAHIT16'][:]
            CIA  = ncid.variables['CIATCCON'][:]

        else:
            CO2   = ncid.variables['CO2'][:]  
            CH4   = ncid.variables['CH4'][:]
            H2O   = ncid.variables['H2O'][:]
        ncid.close()
    
        """
        #CHRIS LUT XSECTIONS + US ATM FILE        
        #US ST. ATM. DATA
        usatmospath = os.path.join(self.calidata,str('AFGLUS_atmos.txt'))
        self.uspressure = np.loadtxt(usatmospath,usecols=0)
        self.usheight = np.loadtxt(usatmospath,usecols=1)
        self.ustemperature = np.loadtxt(usatmospath,usecols=2)
        self.usair = np.loadtxt(usatmospath,usecols=3)
        self.usCH4 = np.loadtxt(usatmospath,usecols=4)
        self.usCO2 = np.loadtxt(usatmospath,usecols=5)
        self.usH2O = np.loadtxt(usatmospath,usecols=6)
        
        ncid = Dataset(self.spectraReffile, 'r')
        specwave        = ncid.variables['Wavelength'][:]
        
        
        if(self.whichBand == 'O2'):
            
            o2path = os.path.join(self.calidata,str('o2_lut_1200-1750nm_0p02fwhm_1e21vcd_mesat.nc'))
            new = Dataset(o2path, 'r')
            TO2   = new.variables['CrossSection'][:,:,:].data  
            Temp = new.variables['Temperature'][:].data
            Press = new.variables['Pressure'][:].data
            Wvl = new.variables['Wavelength'][:].data
            
            
            fn = RegularGridInterpolator((Press,Temp,Wvl), TO2)
            
            wgtO2 = np.zeros(len(Wvl))
            columntotal = 0.0
            for i in range( len(self.usheight) ):
                if(self.usheight[i] <= 58.0):
                    x = fn((self.uspressure[i],self.ustemperature[i],Wvl)) * self.usair[i]
                    wgtO2 = wgtO2 + x
                    columntotal = columntotal + self.usair[i]
            wgtO2 = wgtO2/columntotal
            
           # wgtO2 = fn((1013,288,Wvl)) 
            y = interpolate.interp1d(Wvl,wgtO2)
            O2 = np.zeros(len(specwave))
            for i in range(len(specwave)):
                O2[i] = y(specwave[i])
            CIA  = ncid.variables['CIAO2AIR'][:]
            #CIA  = ncid.variables['CIATCCON'][:]
            ncid.close()

            
        else:
            h2opath = os.path.join(self.calidata,str('h2o_lut_1200-1750nm_0p02fwhm_1e21vcd_mesat.nc'))
            co2path = os.path.join(self.calidata,str('co2_lut_1200-1750nm_0p02fwhm_1e21vcd_mesat.nc'))
            ch4path = os.path.join(self.calidata,str('ch4_lut_1200-1750nm_0p02fwhm_1e17vcd_mesat.nc'))
            
            new = Dataset(h2opath, 'r')
            TH2O   = new.variables['CrossSection'][:,:,:].data  
            Temp = new.variables['Temperature'][:].data
            Press = new.variables['Pressure'][:].data
            self.Wvl = new.variables['Wavelength'][:].data
            new.close()

            new = Dataset(ch4path, 'r')
            TCH4 = new.variables['CrossSection'][:,:,:].data 
            new.close()
            
            new = Dataset(co2path, 'r')
            TCO2 = new.variables['CrossSection'][:,:,:].data  
            new.close()
            
            self.fnCH4 = RegularGridInterpolator((Press,Temp,self.Wvl), TCH4)
            self.fnCO2 = RegularGridInterpolator((Press,Temp,self.Wvl), TCO2)
            self.fnH2O = RegularGridInterpolator((Press,Temp,self.Wvl), TH2O)
            
            wgtCH4 = np.zeros(len(self.Wvl))
            wgtCO2 = np.zeros(len(self.Wvl))
            wgtH2O = np.zeros(len(self.Wvl))
                        
            columntotalCH4 = 0.0
            columntotalCO2 = 0.0
            columntotalH2O = 0.0
            
            if(self.fitSZA):
                
                for i in range( len(self.usheight) ):
                    if(self.usheight[i] <= self.ALT):
                        xCH4 = self.fnCH4((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usCH4[i] * (1.0/np.cos(self.SZA) + 1.0)
                        wgtCH4 = wgtCH4 + xCH4
                        columntotalCH4 = columntotalCH4 + self.usCH4[i]
                        ###
                        xCO2 = self.fnCO2((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usCO2[i] * (1.0/np.cos(self.SZA) + 1.0)
                        wgtCO2 = wgtCO2 + xCO2
                        columntotalCO2 = columntotalCO2 + self.usCO2[i]
                        ###
                        xH2O = self.fnH2O((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usH2O[i] * (1.0/np.cos(self.SZA) + 1.0)
                        wgtH2O = wgtH2O + xH2O
                        columntotalH2O = columntotalH2O + self.usH2O[i]
                    else:
                        xCH4 = self.fnCH4((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usCH4[i] * (1.0/np.cos(self.SZA) )
                        wgtCH4 = wgtCH4 + xCH4
                        columntotalCH4 = columntotalCH4 + self.usCH4[i]
                        ###
                        xCO2 = self.fnCO2((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usCO2[i] * (1.0/np.cos(self.SZA))
                        wgtCO2 = wgtCO2 + xCO2
                        columntotalCO2 = columntotalCO2 + self.usCO2[i]
                        ###
                        xH2O = self.fnH2O((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usH2O[i] * (1.0/np.cos(self.SZA))
                        wgtH2O = wgtH2O + xH2O
                        columntotalH2O = columntotalH2O + self.usH2O[i]
                    
                
                
            else:
                for i in range( len(self.usheight) ):
                    if(self.usheight[i] <= 58.0):
                        xCH4 = self.fnCH4((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usCH4[i]
                        wgtCH4 = wgtCH4 + xCH4
                        columntotalCH4 = columntotalCH4 + self.usCH4[i]
                        ###
                        xCO2 = self.fnCO2((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usCO2[i]
                        wgtCO2 = wgtCO2 + xCO2
                        columntotalCO2 = columntotalCO2 + self.usCO2[i]
                        ###
                        xH2O = self.fnH2O((self.uspressure[i],self.ustemperature[i],self.Wvl)) * self.usH2O[i]
                        wgtH2O = wgtH2O + xH2O
                        columntotalH2O = columntotalH2O + self.usH2O[i]
                    
                    
                    
                    
                    
            wgtCO2 = wgtCO2/columntotalCO2
            wgtCH4 = wgtCH4/columntotalCH4
            wgtH2O = wgtH2O/columntotalH2O
            
            yCH4 = interpolate.interp1d(self.Wvl,wgtCH4)
            CH4 = np.zeros(len(specwave))
            
            yCO2 = interpolate.interp1d(self.Wvl,wgtCO2)
            CO2 = np.zeros(len(specwave))
            
            yH2O = interpolate.interp1d(self.Wvl,wgtH2O)
            H2O = np.zeros(len(specwave))
            
            for i in range(len(specwave)):
                CH4[i] = yCH4(specwave[i])
                CO2[i] = yCO2(specwave[i])
                H2O[i] = yH2O(specwave[i])
                
                
            
            ncid.close()


        ##############################
        
        # DEFINE CROSS TRACK POSITIONS TO FIT
        counter = 0
        
        if(self.whichBand == 'CH4'): 
            headerStr='MethaneAIR_L1B_CH4_'
            l1FitISRFPath = os.path.join(self.l1DataDir,headerStr
                                    +np.min(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                    +np.max(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                    +str('ISRF_proxy_fit_CH4.txt'))
            l1FitPath = os.path.join(self.l1DataDir,headerStr
                                  +np.min(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +np.max(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +str('proxy_fit_CH4.txt'))

            f = open(l1FitPath,"w")
            g = open(l1FitISRFPath,"w")
        else:
            headerStr='MethaneAIR_L1B_O2_'
            l1FitISRFPath = os.path.join(self.l1DataDir,headerStr
                                    +np.min(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                    +np.max(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                    +str('ISRF_proxy_fit_O2.txt'))
            l1FitPath = os.path.join(self.l1DataDir,headerStr
                                  +np.min(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +np.max(frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +str('proxy_fit_O2.txt'))
            f = open(l1FitPath,"w")
            g = open(l1FitISRFPath,"w")
    
    
        #THE FITTING STARTS HERE
        start = 0

        #for j in range (self.pixlimitX[0],self.pixlimitX[1]):
        
        for j in np.arange (self.pixlimitX[0],self.pixlimitX[1],10):
            xTrack = j
            error=False
            
            if (self.whichBand == 'CH4'):
                self.isrf_lut0 = self.isrf_lut[xTrack,:,:]
            else:
                self.isrf_lut0 = self.isrf_lut[xTrack,:,:]
              #  isrf = np.transpose(self.isrf_lut, [2,1,0] )
              #  self.isrf_lut0 = isrf[xTrack,:,:] 
                
            
            #OBSERVED DATA   
                
            wavelength_obs = self.wavelength[xTrack,:].squeeze()
            radiance_obs = rad_obs[xTrack,:].squeeze()
            

            

            if math.isnan( wavelength_obs[self.pixlimitX[0]]):
                print(str(xTrack)+ " has a nan values in wvl")
            elif math.isnan( radiance_obs[self.pixlimitX[0]]):
                print(str(xTrack)+ " has a nan values in rad")
            else:    
                print('FITTING TRACK: '+str(xTrack))
                
                # GET THE GRID OF INTEREST FOR INTERPOLATION
                width = 1*(self.pixlimit[1] - self.pixlimit[0])
                d = np.linspace(wavelength_obs[self.pixlimit[0]],wavelength_obs[self.pixlimit[1]], width+1  )
                
                ##############################
                # DEFINE FITTING VARIABLES:
                ##############################
                params = Parameters()
                #BASELINE CORRECTION
                
                if(start == 0):
                

                    if(self.whichBand == 'CH4'):
                        f0 = Parameter('par_f'+str(0),0.1402161187)
                        params.add(f0)
                        sc_solar = Parameter('scale_solar',0.10873476*self.numframes,min=0)
                        params.add(sc_solar) 
                        f1 = Parameter('baseline0',-0.12)
                        f2 = Parameter('baseline1',8.1097e-05)
                        params.add(f1)
                        params.add(f2)

                    else:
                        f0 = Parameter('par_f'+str(0),0.0421187)
                        params.add(f0)
                        sc_solar = Parameter('scale_solar',0.1893476*self.numframes,min=0)
                        params.add(sc_solar) 
                        f1 = Parameter('baseline0',101.3815)
                        f2 = Parameter('baseline1',-0.159985)
                        f3 = Parameter('baseline2',6.3120e-05)
                        params.add(f1)
                        params.add(f2)
                        params.add(f3)
                        #b1 = Parameter('al0',1.0)
                      #  b2 = Parameter('al1',0.0)
                      #  b3 = Parameter('al2',0)
                       # params.add(b1)
                      #  params.add(b2)
                      #  params.add(b3)


                    if(self.whichBand == 'CH4'):
                        sc_co2 = Parameter('scale_co2',6.7378e+22,max=1e24, min=1)
                        sc_ch4 = Parameter('scale_ch4', 2.9832e+20,max=1e24, min=1)
                        sc_h2o = Parameter('scale_h2o', 1.5151e+23,max=1e25, min=1)
                        params.add(sc_co2)
                        params.add(sc_ch4)
                        params.add(sc_h2o)
                    else:
                        sc_o2 = Parameter('scale_o2',5.8065e+24,max=1e28, min=1)
                        params.add(sc_o2)
                        #sc_cia = Parameter('scale_cia',8e+24,max=1e28, min=1)
                        #params.add(sc_cia)
                        
                        
                                        
                    if(self.fitisrf == True):

                        if(self.ISRF == 'SQUEEZE'):
                           # p1 = Parameter('squeeze',1.0,min=0.6,max=1.3)
                           # params.add(p1)

                            wavelength = np.zeros((len(self.isrf_dw0),len(self.isrf_w) ))
                            self.width = np.ones(len(self.isrf_w))
                            self.sharp = np.ones(len(self.isrf_w))
                            
                            for i in range(len(self.isrf_w)):
                                wavelength[:,i] = self.isrf_w[i] + self.isrf_dw0[:]
                            d_max = np.max(d)
                            d_min = np.min(d)     
                            
                            self.count=0
                            self.index = []
                            for i in range(len(self.isrf_w)):
                                if((d_min < wavelength[-1,i]) and (d_max > wavelength[0,i]) ):
                                    self.index.append(i)
                                    p1 = Parameter('squeeze'+str(self.count),self.width[i],max=1.2,min=0.8)
                                    params.add(p1)
                                    p2 = Parameter('sharp'+str(self.count),self.sharp[i],max=1.2,min=0.8)
                                    params.add(p2)
                                    self.count=self.count+1
                                else:
                                    pass
                        
                        
                        
                        
                        
                        
                        else:  
                            wavelength = np.zeros((len(self.isrf_dw0),len(self.isrf_w) ))
                            self.width = np.zeros(len(self.isrf_w))
                            
                            for i in range(len(self.isrf_w)):
                                wavelength[:,i] = self.isrf_w[i] + self.isrf_dw0[:]
                            d_max = np.max(d)
                            d_min = np.min(d)                  
                                  
                                  
                            if(self.ISRF == 'GAUSS'):
                                for i in range(len(self.isrf_w)):
                                    popt, pcov = optimize.curve_fit(gaussian,self.isrf_dw0,self.isrf_lut0[i,:])
                                    self.width[i] = popt[0]
                                self.count=0
                                self.index = []
                                for i in range(len(self.isrf_w)):
                                    if((d_min < wavelength[-1,i]) and (d_max > wavelength[0,i]) ):
                                        self.index.append(i)
                                        p1 = Parameter('width'+str(self.count),self.width[i],max=1,min=0.01)
                                        params.add(p1)
                                        self.count=self.count+1
                                    else:
                                        pass
                            elif(self.ISRF == 'SUPER'):
                                self.shape = np.zeros(len(self.isrf_w))
                                for i in range(len(self.isrf_w)):
                                    popt, pcov = optimize.curve_fit(supergaussian,self.isrf_dw0,self.isrf_lut0[i,:])
                                    self.width[i] = popt[0]
                                    self.shape[i] = popt[1]
                                self.isrf_super = np.vstack((self.width,self.shape))
                                self.count=0
                                self.index = []
                                for i in range(len(self.isrf_w)):
                                    if((d_min < wavelength[-1,i]) and (d_max > wavelength[0,i]) ):
                                        p1 = Parameter('width'+str(self.count),self.width[i],max=1,min=0.01)
                                        p2 = Parameter('shape'+str(self.count),self.shape[i],max=5,min=0.01)
                                        params.add(p1)
                                        params.add(p2)
                                        self.count=self.count+1
                                        self.index.append(i)
                                    else:
                                        pass
                        
                        

                        
                else:
                    if(self.whichBand == 'CH4'):
                        f0 = Parameter('par_f'+str(0),self.f0CH4)
                        params.add(f0)
                        sc_solar = Parameter('scale_solar',self.solarCH4,min=0)
                        params.add(sc_solar) 
                        f1 = Parameter('baseline0',self.b0CH4)
                        f2 = Parameter('baseline1',self.b1CH4)
                        params.add(f1)
                        params.add(f2)
                        
                        sc_co2 = Parameter('scale_co2',self.scaleco2,max=1e24, min=1)
                        sc_ch4 = Parameter('scale_ch4', self.scalech4,max=1e24, min=1)
                        sc_h2o = Parameter('scale_h2o', self.scaleh2o,max=1e25, min=1)
                        params.add(sc_co2)
                        params.add(sc_ch4)
                        params.add(sc_h2o)

                    else:
                        f0 = Parameter('par_f'+str(0),self.f0O2)
                        params.add(f0)
                        sc_solar = Parameter('scale_solar',self.solarO2,min=0)
                        params.add(sc_solar) 
                        f1 = Parameter('baseline0',self.b0O2)
                        f2 = Parameter('baseline1',self.b1O2)
                        f3 = Parameter('baseline2',self.b2O2)
                        params.add(f1)
                        params.add(f2)
                        params.add(f3)
                        
                        sc_o2 = Parameter('scale_o2',self.scaleo2,max=1e28, min=1)
                        params.add(sc_o2)
                        #sc_cia = Parameter('scale_cia',8e+24,max=1e28, min=1)
                        #params.add(sc_cia)

                        
                        
                                        
                    if(self.fitisrf == True):

                        if(self.ISRF == 'SQUEEZE'):
                            #p1 = self.squeeze
                           # params.add(p1)
                            for i in range(len(self.index)):
                                p1 = Parameter('squeeze'+str(i),self.width[self.index[i]],max=1.2,min=0.8)
                                params.add(p1)
                                p2 = Parameter('sharp'+str(i),self.sharp[self.index[i]],max=1.2,min=0.8)
                                params.add(p2)


                        if(self.ISRF == 'GAUSS'):
                            for i in range(len(self.index)):
                                p1 = Parameter('width'+str(i),self.width[self.index[i]],max=1,min=0.01)
                                params.add(p1)

                        elif(self.ISRF == 'SUPER'):
                            for i in range(len(self.index)):
                                p1 = Parameter('width'+str(i),self.width[self.index[i]],max=1,min=0.01)
                                p2 = Parameter('shape'+str(i),self.shape[self.index[i]],max=5,min=0.01)
                                params.add(p1)
                                params.add(p2)
      


                fit_kws={'xtol':1e-4,'ftol':1e-4}
                ##############################
                # FIT THE DATA
                #first_fit = time.time()
                if(self.whichBand == 'CH4'):
                    lsqFit = minimize( self.spectrumResidual_CH4, params, args=(specwave,CH4,CO2,H2O,\
                    radiance_obs,wavelength_obs,xTrack),method='leastsq',max_nfev=1000,**fit_kws)
                else:
                    lsqFit = minimize( self.spectrumResidual_O2, params, args=(specwave,O2,CIA,\
                    radiance_obs,wavelength_obs,xTrack),method='leastsq',max_nfev=1000,**fit_kws)
                    
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
                
                

                #THIS IS FOR PURPOSES OF PLOTTING END RESULT     

                if(self.fitisrf == True):
                    if(self.ISRF == 'SQUEEZE'):

                        self.isrf_dw0_new = np.zeros((len(self.isrf_w),len(self.isrf_dw0))  )
                        #CREATE NEW dw0 grids
                        for i in range(len(self.isrf_w)):
                            self.isrf_dw0_new[i,:] = self.isrf_dw0
                            
                        for i in range(len(self.index)):
                            self.width[self.index[i]] = params['squeeze'+str(i)].value
                            self.sharp[self.index[i]] = params['sharp'+str(i)].value
                            
                            fwhm0 = self.FWHM(self.isrf_dw0,self.isrf_lut0[self.index[i],:])                  
                            self.isrf_lut0[self.index[i],:] = (self.isrf_lut0[self.index[i],:]) ** self.sharp[self.index[i]]                    
                            fwhm1 = self.FWHM(self.isrf_dw0,self.isrf_lut0[self.index[i],:])
                            stretchfactor = fwhm0/fwhm1   
                            
                            self.isrf_dw0_new[self.index[i],:] = self.isrf_dw0 * self.width[self.index[i]]*stretchfactor
                            
                            
                        ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                        if(self.whichBand=='CH4'):
                            ICH4 = F_isrf_convolve_fft(specwave,CH4,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                            ICO2 = F_isrf_convolve_fft(specwave,CO2,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                            IH2O = F_isrf_convolve_fft(specwave,H2O,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                        else:
                            IO2 = F_isrf_convolve_fft(specwave,O2,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                            ICIA = F_isrf_convolve_fft(specwave,CIA,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)

                            
                            
                            
                    elif( self.ISRF == 'GAUSS'):
                        for i in range(len(self.index)):
                            self.width[self.index[i]] = lsqFit.params['width'+str(i)].value
                        if(self.whichBand=='CH4'):
                            ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                            ICH4 = F_isrf_convolve_fft(specwave,CH4,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                            ICO2 = F_isrf_convolve_fft(specwave,CO2,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                            IH2O = F_isrf_convolve_fft(specwave,H2O,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                        else:
                            ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                            IO2 = F_isrf_convolve_fft(specwave,O2,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                            ICIA = F_isrf_convolve_fft(specwave,CIA,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)

                        
                    else:
                        for i in range(len(self.index)):
                            self.width[self.index[i]] = lsqFit.params['width'+str(i)].value
                            self.shape[self.index[i]] = lsqFit.params['shape'+str(i)].value
                            #UPDATE ISRF WITH NEW SUERGAUSSIAN PARAMETERS
                            self.isrf_super[0,self.index[i]] = self.width[self.index[i]]
                            self.isrf_super[1,self.index[i]] = self.shape[self.index[i]]
                        if(self.whichBand=='CH4'):
                            ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                            ICH4 = F_isrf_convolve_fft(specwave,CH4,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                            ICO2 = F_isrf_convolve_fft(specwave,CO2,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                            IH2O = F_isrf_convolve_fft(specwave,H2O,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                        else:
                            ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                            ICIA = F_isrf_convolve_fft(specwave,CIA,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                            IO2 = F_isrf_convolve_fft(specwave,O2,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                        
                else:
                    ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
                    if(self.whichBand=='CH4'):
                        ICH4 = F_isrf_convolve_fft(specwave,CH4,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
                        ICO2 = F_isrf_convolve_fft(specwave,CO2,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
                        IH2O = F_isrf_convolve_fft(specwave,H2O,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
                    else:
                        IO2 = F_isrf_convolve_fft(specwave,O2,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
                        ICIA = F_isrf_convolve_fft(specwave,CIA,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
                    
        


                #NOW WE NEED TO CREATE THE SHIFTED SPECTRUM: 
                newobs_wavelength = np.zeros(width+1)
                # Set up polynomial coefficients to fit ncol<-->wvl
                p = np.zeros(1)
                for i in range(0,1):
                    p[i] = lsqFit.params['par_f'+str(i)].value
                           
                # UPDATE OBSERVED WAVELENGTHS/SPECTRA
                radnew = np.zeros(width+1)
                idx_finite = np.isfinite(radiance_obs)
                
                IobsFIT = interpolate.splrep(wavelength_obs[idx_finite],radiance_obs[idx_finite])
                Ioriginal = interpolate.splev(d,IobsFIT,der=0)
                
                radinterp = interpolate.interp1d(wavelength_obs[idx_finite],radiance_obs[idx_finite])  
                for i in range(width+1):
                    newobs_wavelength[i] =  d[i] + p[0]
                    
                for i in range(width+1):
                    radnew[i] = radinterp(newobs_wavelength[i])
                       
                
                InewFIT = interpolate.splrep(newobs_wavelength,radnew)
                Inew = interpolate.splev(newobs_wavelength,InewFIT,der=0)



                #CREATE SIMULATED SPECTRA
                ISOLAR = ISOLAR * lsqFit.params['scale_solar'].value
                    
                if(self.whichBand=='CH4'):
                    ICH4 = np.exp(-ICH4 * lsqFit.params['scale_ch4'].value)
                    ICO2 = np.exp(-ICO2 * lsqFit.params['scale_co2'].value ) 
                    IH2O = np.exp(-IH2O * lsqFit.params['scale_h2o'].value )   
                    a0 = lsqFit.params['baseline0'].value
                    a1 = lsqFit.params['baseline1'].value

                    
                    baseline = np.zeros(width+1)
                    for i in range(width+1):
                        baseline[i] = a0 + a1*d[i] 
                        
                else:
                    IO2 = np.exp(-IO2 * lsqFit.params['scale_o2'].value ) 
                    ICIA =  np.exp(-ICIA * (params['scale_o2'].value**2)/1E20 )  
                    
                    a0 = lsqFit.params['baseline0'].value
                    a1 = lsqFit.params['baseline1'].value
                    a2 = lsqFit.params['baseline2'].value
                    
                    #b0 = lsqFit.params['al0'].value
             #       b1 = lsqFit.params['al1'].value
             #       b2 = lsqFit.params['al2'].value
                    
            #        albedo = np.zeros(width+1)
                    
              #      for i in range(width+1):
              #           albedo[i] = 1.0 + (b1 * (d[i] - 1250)) + (b2 * (d[i] - 1250.0)**2) #+ (b2*d[i]*d[i]) 

                    
                    baseline = np.zeros(width+1)
                    for i in range(width+1):
                        baseline[i] = a0 + a1*d[i] + a2*d[i]*d[i] 
                    
                Isim = np.zeros(width+1)
                if(self.whichBand=='CH4'):
                    Isim = (ICH4 + ICO2 + IH2O ) * ISOLAR + baseline 
                else:
                    Isim = (( IO2  +ICIA) *ISOLAR) + baseline
                    
                residual = np.zeros( width+1  )
                residual = 100.0*(Inew - Isim)/Inew
                
                if(self.ISRF == 'GAUSS'):
                
                    #############
                    
                    plt.subplot(211)
                    plt.plot(d,Isim,color='red',label='ISRF Gauss Fit: Simulated')
                    plt.plot((d-p[0]),Ioriginal,color='black',label='Shifted Observation')
                    plt.plot(d,Ioriginal,color='black',linestyle='dotted',label='Original Observation')
                    plt.ylabel('Radiance (1e14 photons/s/cm2/nm/sr)')
                    plt.legend()
                    #############
                    plt.subplot(212)
                    plt.plot(d,residual,color='red',label='Residuals: Obs. - Gauss Sim.')
                    plt.ylabel('Residual (%)')
                    plt.xlabel('Wavelength (nm)')
                    plt.legend()
                    #############
                
                elif(self.ISRF == 'ISRF'):
                    #############
                    plt.subplot(211)
                    plt.plot(d,Isim,color='green',label='ISRF ISRF LUT Fit: Simulated')
                    plt.legend()
                    #############
                    plt.subplot(212)
                    plt.plot(d,residual,color='green',label='Residuals: Obs. - ISRF_LUT Sim.')
                    plt.legend()
                    #############                
                
                elif(self.ISRF == 'SUPER'):
                    #############
                    plt.subplot(211)
                    plt.plot(d,Isim,color='blue',label='ISRF SuperGauss Fit: Simulated')
                    plt.plot((d-p[0]),Ioriginal,color='black',label='Shifted Observation')
                    plt.plot(d,Ioriginal,color='black',linestyle='dotted',label='Original Observation')
                    plt.ylabel('Radiance (1e14 photons/s/cm2/nm/sr)')
                    plt.legend()
                    #############
                    plt.subplot(212)
                    plt.plot(d,residual,color='blue',label='Residuals: Obs. - SuperGauss Sim.')
                    plt.legend()
                    #############                   
                elif(self.ISRF == 'SQUEEZE'):
                    #############
                    plt.subplot(211)
                    plt.plot((d-p[0]),Ioriginal,color='black',label='Shifted Observation')
                    plt.plot(d,Ioriginal,color='black',linestyle='dotted',label='Original Observation')
                    plt.ylabel('Radiance (1e14 photons/s/cm2/nm/sr)')
                    plt.plot(d,Isim,color='orange',label='ISRF Squeeze Fit: Simulated')
                    plt.legend()
                    #############
                    plt.subplot(212)
                    plt.plot(d,residual,color='orange',label='Residuals: Obs. - Squeeze Sim.')
                    plt.legend()
                
                
               # plt.show()
              #  plt.subplot(313)
              #  plt.plot(d,ISOLAR,label='Solar')
              #  plt.ylabel('Radiance (1e14 photons/s/cm2/nm/sr)')
             #   plt.xlabel('Wavelength (nm)')
             #   plt.legend()
                #plt.show()
                return






                if(self.whichBand == 'O2'):
                    if((lsqFit.params['scale_o2'].stderr == None) or (lsqFit.params['scale_solar'].stderr == None) or \
                    (lsqFit.params['par_f'+str(0)].stderr == None) or (lsqFit.params['par_f'+str(0)].value == 0) ):
                        error=True
                elif(self.whichBand == 'CH4'):
                    if((lsqFit.params['scale_co2'].stderr == None) or (lsqFit.params['scale_solar'].stderr == None) or \
                     (lsqFit.params['par_f'+str(0)].stderr == None) or (lsqFit.params['par_f'+str(0)].value == 0) ):
                         error=True
                else:
                    pass
                
                
                
                
                if(self.whichBand == 'CH4' and error == False):
                    errorch4 = 100*lsqFit.params['scale_ch4'].stderr/lsqFit.params['scale_ch4'].value
                    errorco2 = 100*lsqFit.params['scale_co2'].stderr/lsqFit.params['scale_co2'].value
                    errorh2o = 100*lsqFit.params['scale_h2o'].stderr/lsqFit.params['scale_h2o'].value
                    errorsolar = 100*lsqFit.params['scale_solar'].stderr/lsqFit.params['scale_solar'].value
                    errorf0 = 100*lsqFit.params['par_f'+str(0)].stderr/lsqFit.params['par_f'+str(0)].value
                    errorb0CH4=100.0*lsqFit.params['baseline0'].stderr/lsqFit.params['baseline0'].value
                    errorb1CH4=100.0*lsqFit.params['baseline1'].stderr/lsqFit.params['baseline1'].value
                elif(self.whichBand == 'O2' and error == False):
                    erroro2 = 100*lsqFit.params['scale_o2'].stderr/lsqFit.params['scale_o2'].value
                    #errorcia = 100*lsqFit.params['scale_cia'].stderr/lsqFit.params['scale_cia'].value
                    errorsolar = 100*lsqFit.params['scale_solar'].stderr/lsqFit.params['scale_solar'].value
                    errorf0 = 100*lsqFit.params['par_f'+str(0)].stderr/lsqFit.params['par_f'+str(0)].value
                    errorb0o2=100.0*lsqFit.params['baseline0'].stderr/lsqFit.params['baseline0'].value
                    errorb1o2=100.0*lsqFit.params['baseline1'].stderr/lsqFit.params['baseline1'].value
                    errorb2o2=100.0*lsqFit.params['baseline2'].stderr/lsqFit.params['baseline2'].value
                else:
                    pass
                
                widtherr = np.zeros(len(self.index))  
                print(len(self.index))  
                for i in range(len(self.index)):
                     if(lsqFit.params['width'+str(i)].stderr == None):
                          pass
                     else:
                          widtherr[i] = 100.0*lsqFit.params['width'+str(i)].stderr/lsqFit.params['width'+str(i)].value
               
                if(self.whichBand == 'CH4'):
                     if( (abs(errorch4) >= 100.0) or (abs(errorco2) >= 100.0) or (abs(errorh2o) >= 100.0) or (abs(errorf0) >= 100.0) or (abs(errorsolar) >= 100.0) ):
                          error=True
                elif(self.whichBand == 'O2'):
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
                        
                        if(self.whichBand == 'CH4'):
                            self.f0CH4=lsqFit.params['par_f'+str(0)].value
                            self.solarCH4=lsqFit.params['scale_solar'].value
                            self.b0CH4=lsqFit.params['baseline0'].value
                            self.b1CH4=lsqFit.params['baseline1'].value
                            self.scaleco2=lsqFit.params['scale_co2'].value
                            self.scalech4=lsqFit.params['scale_ch4'].value
                            self.scaleh2o=lsqFit.params['scale_h2o'].value


                        else:
                            self.f0O2=lsqFit.params['par_f'+str(0)].value
                            self.solarO2=lsqFit.params['scale_solar'].value
                            self.b0O2=lsqFit.params['baseline0'].value
                            self.b1O2=lsqFit.params['baseline1'].value
                            self.b2O2=lsqFit.params['baseline2'].value
                            self.scaleo2=lsqFit.params['scale_o2'].value
                            
                        if(self.fitisrf == True):

                            if(self.ISRF == 'SQUEEZE'):
                                for i in range(len(self.index)):
                                    self.width[self.index[i]] = lsqFit.params['squeeze'+str(i)].value
                                    self.sharp[self.index[i]] = lsqFit.params['sharp'+str(i)].value

                            if(self.ISRF == 'GAUSS'):
                                for i in range(len(self.index)):
                                    self.width[self.index[i]] = lsqFit.params['width'+str(i)].value


                            elif(self.ISRF == 'SUPER'):
                                for i in range(len(self.index)):
                                    self.width[self.index[i]] = lsqFit.params['width'+str(i)].value
                                    self.shape[self.index[i]] = lsqFit.params['shape'+str(i)].value



                        
                        if(self.whichBand == 'CH4'):
                            f.write(str(xTrack)+" "+str(lsqFit.params['par_f'+str(0)].value)+" "+str(errorf0) +" "+ \
                            str(lsqFit.params['scale_solar'].value)+" "+str(errorsolar)+" "+str(lsqFit.params['scale_ch4'].value)+" "+\
                            str(errorch4)+" "+str(lsqFit.params['scale_co2'].value)+" "+str(errorco2) + " " + \
                            str(lsqFit.params['scale_h2o'].value)+" "+str(errorh2o) + " " + \
                            str(lsqFit.params['baseline0'].value) +" "+ str(errorb0CH4) + " " +\
                            str(lsqFit.params['baseline1'].value) +" "+ str(errorb1CH4) + "\n")
                            if(self.fitisrf == True):
                                g.write(str(xTrack)+" ")
                                for i in range(len(self.index)):
                                    g.write(str(self.index[i]) + " "+str(self.width[self.index[i]]) + " "+ str(widtherr[i])+" "    )
                                g.write("\n")
                        else:
                            f.write(str(xTrack)+" "+str(lsqFit.params['par_f'+str(0)].value)+" "+str(errorf0) +" "+ \
                            str(lsqFit.params['scale_solar'].value)+" "+str(errorsolar)+" " + \
                            str(lsqFit.params['scale_o2'].value)+" "+str(erroro2) + " " + \
                            str(lsqFit.params['baseline0'].value) +" "+ str(errorb0o2) + " " +\
                            str(lsqFit.params['baseline1'].value) +" "+ str(errorb1o2) + " " +\
                            str(lsqFit.params['baseline2'].value) +" "+ str(errorb2o2) + "\n")
                            #+str(lsqFit.params['scale_cia'].value)+" "+str(errorcia)+" "\
                            if(self.fitisrf == True):
                                g.write(str(xTrack)+" ")
                                for i in range(len(self.index)):
                                    g.write(str(self.index[i]) + " "+str(self.width[self.index[i]]) + " "+ str(widtherr[i])+" ")
                                g.write("\n")

        g.close()
        f.close()


    def spectrumResidual_CH4(self,params,specwave,CH4,CO2,H2O,radiance_obs,wavelength_obs,xTrack):
        from scipy import interpolate,optimize
        # WAVELENGTH GRID OF INTEREST
        width = self.pixlimit[1] - self.pixlimit[0]
        d = np.linspace(wavelength_obs[self.pixlimit[0]],wavelength_obs[self.pixlimit[1]], width+1  )
        # CONVOLVED SPECTRA IN THE INTERVAL D[:]

        a0 = params['baseline0'].value
        a1 = params['baseline1'].value
        
        
        baseline = np.zeros(width+1)
        
        for i in range(width+1):
             baseline[i] = a0 + (a1 * d[i]) 
        

        if(self.fitisrf == True):
        
        
            if( self.ISRF == 'SQUEEZE'):
            
                
                self.isrf_dw0_new = np.zeros((len(self.isrf_w),len(self.isrf_dw0))  )
                #CREATE NEW dw0 grids
                for i in range(len(self.isrf_w)):
                    self.isrf_dw0_new[i,:] = self.isrf_dw0
                for i in range(len(self.index)):
                    self.width[self.index[i]] = params['squeeze'+str(i)].value
                    self.sharp[self.index[i]] = params['sharp'+str(i)].value
                    
                    fwhm0 = self.FWHM(self.isrf_dw0,self.isrf_lut0[self.index[i],:])                  
                    self.isrf_lut0[self.index[i],:] = (self.isrf_lut0[self.index[i],:])** self.sharp[self.index[i]]                    
                    fwhm1 = self.FWHM(self.isrf_dw0,self.isrf_lut0[self.index[i],:])
                    stretchfactor = fwhm0/fwhm1   
                    self.isrf_dw0_new[self.index[i],:] = self.isrf_dw0 * self.width[self.index[i]]*stretchfactor
                    
                    
                ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                ICH4 = F_isrf_convolve_fft(specwave,CH4,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                ICO2 = F_isrf_convolve_fft(specwave,CO2,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                IH2O = F_isrf_convolve_fft(specwave,H2O,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)

                
                
                
                
            elif( self.ISRF == 'GAUSS'):
                for i in range(len(self.index)):
                    self.width[self.index[i]] = params['width'+str(i)].value

                ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                ICH4 = F_isrf_convolve_fft(specwave,CH4,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                ICO2 = F_isrf_convolve_fft(specwave,CO2,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
                IH2O = F_isrf_convolve_fft(specwave,H2O,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
            else:

                for i in range(len(self.index)):
                    self.width[self.index[i]] = params['width'+str(i)].value
                    self.shape[self.index[i]] = params['shape'+str(i)].value
                    #UPDATE ISRF WITH NEW SUERGAUSSIAN PARAMETERS
                    self.isrf_super[0,self.index[i]] = self.width[self.index[i]]
                    self.isrf_super[1,self.index[i]] = self.shape[self.index[i]]
                

                ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                ICH4 = F_isrf_convolve_fft(specwave,CH4,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                ICO2 = F_isrf_convolve_fft(specwave,CO2,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                IH2O = F_isrf_convolve_fft(specwave,H2O,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
            
            
            
        else:
            ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
            ICH4 = F_isrf_convolve_fft(specwave,CH4,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
            ICO2 = F_isrf_convolve_fft(specwave,CO2,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
            IH2O = F_isrf_convolve_fft(specwave,H2O,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)



        #NOW WE NEED TO CREATE THE SHIFTED SPECTRUM: 
        newobs_wavelength = np.zeros(width+1)
        # Set up polynomial coefficients to fit ncol<-->wvl
        p = np.zeros(1)
        for i in range(0,1):
            p[i] = params['par_f'+str(i)].value
            
        # UPDATE OBSERVED WAVELENGTHS/SPECTRA
        radnew = np.zeros(width+1)
        idx_finite = np.isfinite(radiance_obs)
        radinterp = interpolate.interp1d(wavelength_obs[idx_finite],radiance_obs[idx_finite])  
        Ioriginal = np.zeros(width+1)
        for i in range(width+1):
            newobs_wavelength[i] =  d[i] + p[0]
            radnew[i] = radinterp(newobs_wavelength[i])
            Ioriginal[i] = radinterp(d[i])
            

        InewFIT = interpolate.splrep(newobs_wavelength,radnew)
        Inew = interpolate.splev(newobs_wavelength,InewFIT,der=0)
        

        #CREATE SIMULATED SPECTRA
        ISOLAR = ISOLAR * params['scale_solar'].value
        ICH4 = np.exp(-ICH4 * params['scale_ch4'].value)
        ICO2 = np.exp(-ICO2 * params['scale_co2'].value )   
        IH2O = np.exp(-IH2O * params['scale_h2o'].value )       

        Isim = np.zeros(width+1)
        Isim = (ICH4 + ICO2 + IH2O ) * ISOLAR + baseline

        residual = np.zeros(width + 1)
        residual = Isim - Inew
        
        return ( residual)

    def spectrumResidual_O2(self,params,specwave,O2,CIA,radiance_obs,wavelength_obs,xTrack):
       from scipy import interpolate,optimize

       width = 1*(self.pixlimit[1] - self.pixlimit[0])
       d = np.linspace(wavelength_obs[self.pixlimit[0]],wavelength_obs[self.pixlimit[1]], width+1  )
       
       
       a0 = params['baseline0'].value
       a1 = params['baseline1'].value
       a2 = params['baseline2'].value
       
       
     #  b1 = params['al1'].value
     #  b2 = params['al2'].value
       
       
      # albedo = np.zeros(width+1)
       baseline = np.zeros(width+1)
       
       for i in range(width+1):
         #   albedo[i] =  1.0 + (b1 * (d[i] - 1250)) + (b2 * (d[i] - 1250.0)**2) 
            baseline[i] = a0 + (a1 * d[i]) + (a2*d[i]*d[i]) 
            

       if(self.fitisrf == True):
       
       
           if( self.ISRF == 'SQUEEZE'):
                self.isrf_dw0_new = np.zeros((len(self.isrf_w),len(self.isrf_dw0))  )
                #CREATE NEW dw0 grids
                for i in range(len(self.isrf_w)):
                    self.isrf_dw0_new[i,:] = self.isrf_dw0
                    
                for i in range(len(self.index)):
                    self.width[self.index[i]] = params['squeeze'+str(i)].value
                    self.sharp[self.index[i]] = params['sharp'+str(i)].value
                    
                    fwhm0 = self.FWHM(self.isrf_dw0,self.isrf_lut0[self.index[i],:])                  
                    self.isrf_lut0[self.index[i],:] = (self.isrf_lut0[self.index[i],:])** self.sharp[self.index[i]]                    
                    fwhm1 = self.FWHM(self.isrf_dw0,self.isrf_lut0[self.index[i],:])
                    stretchfactor = fwhm0/fwhm1   
                    self.isrf_dw0_new[self.index[i],:] = self.isrf_dw0 * self.width[self.index[i]]*stretchfactor
                    
                ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                IO2 = F_isrf_convolve_fft(specwave,O2,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)
                ICIA = F_isrf_convolve_fft(specwave,CIA,d,self.isrf_w,self.isrf_dw0_new,self.isrf_lut0,self.ISRF,self.fitisrf)

               
               
           elif( self.ISRF == 'GAUSS'):
               for i in range(len(self.index)):
                   self.width[self.index[i]] = params['width'+str(i)].value

               ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
               IO2 = F_isrf_convolve_fft(specwave,O2,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
               ICIA = F_isrf_convolve_fft(specwave,CIA,d,self.isrf_w,self.isrf_dw0,self.width,self.ISRF,self.fitisrf)
               
           else:
                for i in range(len(self.index)):
                    self.width[self.index[i]] = params['width'+str(i)].value
                    self.shape[self.index[i]] = params['shape'+str(i)].value
                    #UPDATE ISRF WITH NEW SUERGAUSSIAN PARAMETERS
                    self.isrf_super[0,self.index[i]] = self.width[self.index[i]]
                    self.isrf_super[1,self.index[i]] = self.shape[self.index[i]]
                

                ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                ICIA = F_isrf_convolve_fft(specwave,CIA,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                IO2 = F_isrf_convolve_fft(specwave,O2,d,self.isrf_w,self.isrf_dw0,self.isrf_super,self.ISRF,self.fitisrf)
                

       else:
           ISOLAR = F_isrf_convolve_fft(self.refWavelength,self.refSolarIrradiance,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
           IO2 = F_isrf_convolve_fft(specwave,O2,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)
           ICIA = F_isrf_convolve_fft(specwave,CIA,d,self.isrf_w,self.isrf_dw0,self.isrf_lut0,self.ISRF,self.fitisrf)  
           


       #NOW WE NEED TO CREATE THE SHIFTED SPECTRUM: 
       newobs_wavelength = np.zeros(width+1)
       # Set up polynomial coefficients to fit ncol<-->wvl
       p = np.zeros(1)
       for i in range(0,1):
           p[i] = params['par_f'+str(i)].value
           
       # UPDATE OBSERVED WAVELENGTHS/SPECTRA
       radnew = np.zeros(width+1)
       idx_finite = np.isfinite(radiance_obs)
       IobsFIT = interpolate.splrep(wavelength_obs[idx_finite],radiance_obs[idx_finite])
       Ioriginal = interpolate.splev(d,IobsFIT,der=0)
       radinterp = interpolate.interp1d(wavelength_obs[idx_finite],radiance_obs[idx_finite])  
       for i in range(width+1):
           newobs_wavelength[i] =  d[i] + p[0]
       
       for i in range(width+1):
                radnew[i] = radinterp(newobs_wavelength[i])
                
       InewFIT = interpolate.splrep(newobs_wavelength,radnew)
       Inew = interpolate.splev(newobs_wavelength,InewFIT,der=0)



       #CREATE SIMULATED SPECTRA
       ISOLAR = ISOLAR * params['scale_solar'].value
       IO2 = np.exp(-IO2 * params['scale_o2'].value )       
       ICIA = np.exp(-ICIA * (params['scale_o2'].value**2)/1E20 )  #np.exp(-ICIA* params['scale_cia'].value )         

       
       Isim = np.zeros(width+1)
       Isim = (( IO2 + ICIA ) *ISOLAR) + baseline
           
       residual = np.zeros( width+1  )

       residual =  Isim - Inew
       

       return ( residual)
   
    
   
    def FWHM(self,x,y):
        hmx = self.half_max_x(x,y)
        fwhm = hmx[1] - hmx[0]
        return fwhm
    
    def lin_interp(self,x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
    
    def half_max_x(self,x, y):
        half = max(y)/2.0
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        return [self.lin_interp(x, y, zero_crossings_i[0], half),
                self.lin_interp(x, y, zero_crossings_i[1], half)]
  

