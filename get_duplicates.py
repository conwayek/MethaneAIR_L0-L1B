import numpy as np
import os
from datetime import datetime

"""This code is really just for the large scale batch processing and resolves some duplicate file issues from Odyssey jobs canceling and re-running etc
Not needed for cloud processing - please ignore.
"""


listdir = []
listdir.append('/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02_V2/O2_NATIVE/')
listdir.append('/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02_V2/O2_15x3/')
listdir.append('/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02_V2/O2_5x1/')
listdir.append('/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02_V2/CH4_NATIVE/')
listdir.append('/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02_V2/CH4_15x3/')
listdir.append('/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF02_V2/CH4_5x1/')

for i in range(len(listdir)):
    direct = listdir[i]

    list_abs_files = []
    tzero_stamp_hr = []
    tzero_stamp_min = []
    tzero_stamp_sec = []
    tend_stamp_hr = []
    tend_stamp_min = []
    tend_stamp_sec = []
    tproc_stamp = []
    
    tzero = []
    tend = []
    tproc = []
    for file in os.listdir(direct):
        if(file.endswith('.nc')):
            list_abs_files.append(file)
            t=file.split(".nc")[0]
            if('O2' in direct):
                t=t.split("MethaneAIR_L1B_O2_")[1]
            else:
                t=t.split("MethaneAIR_L1B_CH4_")[1]
            t=t.split("_")
            tzero.append( datetime.strptime(t[0],"%Y%m%dT%H%M%S"))
            tend.append( datetime.strptime(t[1],"%Y%m%dT%H%M%S"))
            tproc.append( datetime.strptime(t[1],"%Y%m%dT%H%M%S"))
    tzero=np.array(tzero)
    tend=np.array(tend)
    tproc=np.array(tproc)
    
    
    nfiles = len(list_abs_files)
    
    caught = np.zeros(nfiles,dtype=np.int8)
    
    
    for i in range(nfiles):
        for j in range(nfiles):
            if( (tzero[i] == tzero[j] ) and (tend[i]==tend[j] ) and (caught[j] ==0 )and (caught[i] ==0 ) and (i!=j)):
                caught[i] = 1
                print(list_abs_files[i],' ==  ',list_abs_files[j] )
                if(tproc[i]>tproc[j]):
                    caught[j] = 1
                    print('Removing ',list_abs_files[j])
                    os.remove(os.path.join(direct,list_abs_files[j]))
                else:
                    caught[i] = 1
                    print('Removing ',list_abs_files[i])
                    os.remove(os.path.join(direct,list_abs_files[i]))
                #os.remove(os.path.join(direct,list_abs_files[i]))
                #exit()


