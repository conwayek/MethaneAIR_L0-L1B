# /bin/bash  
# ----------------Parameters---------------------- #            
#$ -S /bin/bash
#$ -t 1-5                                                
#$ -pe mthread 1                                               
#$ -q sThM.q                                                   
#$ -l mres=50G,h_data=50G,h_vmem=50G                           
#$ -l himem                                                    
#$ -cwd                                                        
#$ -j y                                                        
#$ -N o2-$TASK_ID                                                   
#$ -o o2.$TASK_ID.log                                                   
#$ -m ea
#$ -M eamon.conway@cfa.harvard.edu                                                     
#                                                              
# ----------------Modules------------------------- #           
unset R_LIBS_SITE   
#                                                              
# ----------------Your Commands------------------- #           
#                                                              
export OMP_NUM_THREADS=$NSLOTS                                 
echo + `date` $JOB_NAME started on $HOSTNAME in $QUEUE with jobID=$JOB_ID and taskID=$SGE_TASK_ID
echo + NSLOTS = $NSLOTS                                        
#                                                              
# Run simulation' 
#                                                              
echo = `date` job $JOB_NAME done                               


###############
# Make the directory be the loop/job index
###############
mkdir -p "${SGE_TASK_ID}"

###############
# Echo the job index to iteration.txt 
###############
echo "${SGE_TASK_ID}" > iteration.txt

###############
# Copy important files to the directory
###############

cp dem_maker.py wavecal_routines.py akaze_nc_06_24_2021.py akaze_nc_ch4_o2_06_24_2021.py darkfinder.py "${SGE_TASK_ID}" 
cp job.sh aggregate.py iteration.txt Orthorectification_Avionics_NC_Fast_Hydra.R Orthorectification_Optimized_NC_Fast_O2_Hydra.R "${SGE_TASK_ID}"
cp Orthorectification_Optimized_NC_Fast_CH4_Hydra.R   MethaneAIR_L1_EKC_07_08_2021.py run_L1B_07_07.py "${SGE_TASK_ID}"


cd "${SGE_TASK_ID}"
python run_L1B_07_07.py 
cd ..

