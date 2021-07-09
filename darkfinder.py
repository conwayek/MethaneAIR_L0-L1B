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
    

def main(direct):

    #l0DataDirCH4 = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/data/flight_testing/20191112/ch4_camera/'
    #l0DataDirO2 = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/data/flight_testing/20191112/o2_camera/'

    files = os.listdir(direct)
    nfiles = len(files)
    dark_files = []
    
    for i in range(nfiles):
        if('.seq' in files[i]):
            data = ReadSeq(os.path.join(direct,files[i]))
            if (np.nanmean(data) < 1900):
                dark_files.append(os.path.join(direct,files[i]))
                #print(files[i],' ',np.nanmean(data))
    return(dark_files)

