import numpy as np
import sys, os
from sys import exit
import datetime as dt
import struct

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
      
    
    return [data]

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
    Seq.ImageSizeBytes = temp[4]
    Seq.NumFrames = 1#temp[6]
    Seq.TrueImageSize = temp[8]
    
    ## Read raw frames
    rawframes = [fin.read(Seq.TrueImageSize) for i in range(Seq.NumFrames)]
    fin.close()
    
    ## Process each frame -- dropping filler bytes before passing raw
    # print('Reading {} frames'.format(Seq.NumFrames))
    frames = [ParseFrame(i,Seq.ImageHeight,Seq.ImageWidth,
                         rawframes[i][:Seq.ImageSizeBytes+6]) \
                    for i in range(Seq.NumFrames)]
    
    Data = np.array([dd[0] for dd in frames])
    
    return(Data)
# above ReadSeq routines are from Jonathan Franklin
    

def main(direct,flight):


    files = os.listdir(direct)
    nfiles = len(files)
    dark_files = []
    
    for i in range(nfiles):
        if('.seq' in files[i]):
            if(flight =='RF04'):
                # These are the five 100 ms exposure darks for RF04. There are several 50 ms exposures we want to exclude from being captured. 
                if(('14_35_27' in files[i]) or ('15_27_27' in files[i] ) or ('17_00_09' in files[i]) or ('18_24_29' in files[i]) or ('19_11_55' in files[i])   ):
                    data = ReadSeq(os.path.join(direct,files[i]))
                    if ((np.nanmean(data) < 1900)):
                        dark_files.append(os.path.join(direct,files[i]))
            elif(flight =='CheckOut'):
                # These are the four 100 ms exposure darks for RF03/CheckOut. There are several 50 ms exposures we want to exclude from being captured.
                if(('17_18_10' in files[i]) or ('18_33_51' in files[i] ) or ('19_39_32' in files[i]) or ('19_58_00' in files[i])   ):
                    data = ReadSeq(os.path.join(direct,files[i]))
                    if ((np.nanmean(data) < 1900)):
                        dark_files.append(os.path.join(direct,files[i]))
            elif(flight =='RF05'):
                if(('14_53_23' in files[i]) or ('15_38_51' in files[i] ) or ('17_35_50' in files[i]) or ('19_23_06' in files[i]) or ('20_52_05' in files[i])  ):
                    data = ReadSeq(os.path.join(direct,files[i]))
                    if ((np.nanmean(data) < 1900)):
                        dark_files.append(os.path.join(direct,files[i]))
            elif(flight =='RF06'):
                if(('14_57_31' in files[i]) or ('16_04_31' in files[i] ) or ('18_31_23' in files[i])  ):
                    data = ReadSeq(os.path.join(direct,files[i]))
                    if ((np.nanmean(data) < 1900)):
                        dark_files.append(os.path.join(direct,files[i]))
            elif(flight =='RF07'):
                if(('14_39_28' in files[i]) or ('15_29_56' in files[i] ) or ('18_04_29' in files[i]) or ('20_30_57' in files[i]) or ('22_02_03' in files[i])  ):
                    data = ReadSeq(os.path.join(direct,files[i]))
                    if ((np.nanmean(data) < 1900)):
                        dark_files.append(os.path.join(direct,files[i]))
            elif(flight =='RF08'):
                if(('14_55_11' in files[i]) or ('15_46_38' in files[i] ) or ('17_01_01' in files[i]) or ('17_36_32' in files[i]) or ('19_42_54' in files[i])  ):
                    data = ReadSeq(os.path.join(direct,files[i]))
                    if ((np.nanmean(data) < 1900)):
                        dark_files.append(os.path.join(direct,files[i]))
            elif(flight =='RF09'):
                if(('14_48_38' in files[i]) or ('16_04_13' in files[i] ) or ('20_07_17' in files[i]) or ('20_56_40' in files[i])  ):
                    data = ReadSeq(os.path.join(direct,files[i]))
                    if ((np.nanmean(data) < 1900)):
                        dark_files.append(os.path.join(direct,files[i]))
            elif(flight =='RF10'):
                if(('22_51_54' in files[i]) or ('00_24_59' in files[i] ) or ('01_12_47' in files[i])):
                    data = ReadSeq(os.path.join(direct,files[i]))
                    if ((np.nanmean(data) < 1900)):
                        dark_files.append(os.path.join(direct,files[i]))
            else:
                print('Darkfile_finder code not setup for this flight - ',str(flight) ,' check')
                exit() 
    return(dark_files)

