import os
import datetime as dt
import math
import numpy as np

#fileo2 = '/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/CheckOut/O2_NATIVE/log_file.txt'
fileo2 = '/n/holylfs04/LABS/wofsy_lab/Lab/econway/DATA/priority_o2_seq_files_RF06.txt'
f = open(fileo2,'r')
data = f.readlines()
f.close()
fileso2 = []
for i in range(len(data)):
    fileso2.append(os.path.basename(data[i].split('\n')[0]))


listch4 = os.listdir('/scratch/sao_atmos/econway/RF06/L0_DATA_RF06/ch4_camera/')

process_files_ch4 = []

donech4 = np.zeros(len(listch4))

for i in range(len(fileso2)):
   to2 = str(fileso2[i]).split('o2_camera_')[1]
   to2_start = to2.split('.seq')[0]
   to2_start =  (dt.datetime.strptime(to2_start,'%Y_%m_%d_%H_%M_%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
   to2_end =  to2_start + 30#dt.timedelta(seconds=30)

   for j in range(len(listch4)):
       tch4 = str(listch4[j]).split('_camera_')[1]
       tch4_start = (dt.datetime.strptime(tch4.strip('.seq'),'%Y_%m_%d_%H_%M_%S') - dt.datetime(1985,1,1,0,0,0)).total_seconds()
       tch4_end = tch4_start + 30#dt.timedelta(seconds=30)        
       if( math.isclose(to2_start,tch4_start,abs_tol=1.4) and math.isclose(to2_end, tch4_end,abs_tol=1.1) and (donech4[j] == 0) ):
           process_files_ch4.append(listch4[j]) 
           donech4[j] = 1
        

f = open('priority_ch4_seq_files_RF06.txt','w')
for i in range(len(process_files_ch4)):
    f.write(str(os.path.join('/scratch/sao_atmos/econway/RF06/L0_DATA_RF06/ch4_camera/',process_files_ch4[i]))+'\n'  )
        
f.close()
        
