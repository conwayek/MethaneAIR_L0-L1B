#!/bin/bash

input="o2_seq_files_RF02.txt"
#input="test.txt"

i=0
while IFS= read -r line

do
echo $i > iteration.txt


mkdir -p "$line"



echo '# /bin/bash  '            >job.sh
echo '# ----------------Parameters---------------------- #             '            >>job.sh
echo '#$ -S /bin/bash                                                  '            >>job.sh
echo '#$ -pe mthread 1                                                 '            >>job.sh
echo '#$ -q sThM.q                                                     '          >>job.sh
echo '#$ -l mres=40G,h_data=40G,h_vmem=40G                             '            >>job.sh
echo '#$ -l himem                                                      '            >>job.sh
echo '#$ -cwd                                                          '            >>job.sh
echo '#$ -j y                                                          '>>job.sh
echo "#$ -N o2-$i                                                  "      >>job.sh
echo '#$ -o o2.log                                                    '            >>job.sh
echo '#$ -m bea                                                        '            >>job.sh
echo '#                                                                '            >>job.sh
echo '# ----------------Modules------------------------- #             '            >>job.sh
echo 'unset R_LIBS_SITE   ' >> job.sh
echo ' ' >> job.sh
echo '#                                                                '            >>job.sh
echo '# ----------------Your Commands------------------- #             '            >>job.sh
echo '#                                                                '            >>job.sh
echo 'export OMP_NUM_THREADS=$NSLOTS                                   '            >>job.sh
echo 'echo + `date` job $JOB_NAME started in $QUEUE with jobID=$JOB_ID on $HOSTNAME '>>job.sh
echo 'echo + NSLOTS = $NSLOTS                                         '            >>job.sh
echo '#                                                                '            >>job.sh
echo '# Run simulation' >> job.sh
echo 'python run_L1B_07_07.py '  >> job.sh
echo '#                                                                '            >>job.sh
echo 'echo = `date` job $JOB_NAME done                                 '            >>job.sh


cp dem_maker.py wavecal_routines.py akaze_nc_06_24_2021.py akaze_nc_ch4_o2_06_24_2021.py darkfinder.py  job.sh aggregate.py iteration.txt Orthorectification_Avionics_NC_Fast_Hydra.R Orthorectification_Optimized_NC_Fast_O2_Hydra.R Orthorectification_Optimized_NC_Fast_CH4_Hydra.R   MethaneAIR_L1_EKC_07_08_2021.py run_L1B_07_07.py "$line"


cd "$line"

#sbatch job.sh
qsub job.sh

cd ..

i=$(( $i + 1 ))
done < "$input"

