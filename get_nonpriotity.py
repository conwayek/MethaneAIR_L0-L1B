import os
import datetime as dt
import math
import numpy as np

file_priority = '/n/holylfs04/LABS/wofsy_lab/Lab/econway/DATA/priority_ch4_seq_files_RF05.txt'
f = open(file_priority,'r')
data = f.readlines()
f.close()
files_priority = []
for i in range(len(data)):
    files_priority.append(os.path.basename(data[i].split('\n')[0]))


full_list = os.listdir('/n/wofsy_lab/MethaneAIR/flight_data/20210803/ch4_camera/')

nonpriority = []

for i in range(len(full_list)):
   tfull = str(full_list[i]).split('ch4_camera_')[1]

   done=False
   for j in range(len(files_priority)):
       tpriority = str(files_priority[j]).split('_camera_')[1]
       if(tfull == tpriority ) :
           done = True
   if(done == False):
       nonpriority.append(full_list[i]) 

f = open('non_priority_ch4_seq_files_RF05.txt','w')
for i in range(len(nonpriority)):
    f.write(nonpriority[i]+'\n'  )
        
f.close()
        
