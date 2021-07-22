#!/bin/bash
#SBATCH -t 0-0:10
#SBATCH -N 1
#SBATCH --job-name=test-array 
#SBATCH --mem-per-cpu=500
#SBATCH -o process_%A_%a.out # Standard output
#SBATCH -e process_%A_%a.err # Standard error
module load R_packages/4.0.5-fasrc01
module load udunits/2.2.26-fasrc02  
module load geos/3.9.1-fasrc01  
module load proj/8.0.0-fasrc01  
module load gdal/3.2.2-fasrc01  
unset R_LIBS_SITE   
export R_LIBS_USER=/n/home04/econway/apps/R_4.0.2

###############
# Make the directory be the loop/job index
###############
mkdir -p "${SLURM_ARRAY_TASK_ID}"

###############
# Echo the job index to iteration.txt 
###############
echo "${SLURM_ARRAY_TASK_ID}" > iteration.txt

###############
# Copy important files to the directory
###############
cp wavecal_routines.py dem_maker.py akaze_nc_06_24_2021.py akaze_nc_ch4_o2_06_24_2021.py  job.sh aggregate.py iteration.txt "${SLURM_ARRAY_TASK_ID}"
cp Orthorectification_Avionics_NC_Fast.R Orthorectification_Optimized_NC_Fast_O2.R "${SLURM_ARRAY_TASK_ID}"
cp Orthorectification_Optimized_NC_Fast_CH4.R   MethaneAIR_L1_EKC_07_08_2021.py run_L1B_07_07.py "${SLURM_ARRAY_TASK_ID}"


cd "${SLURM_ARRAY_TASK_ID}"

python run_L1B_07_07.py 
#echo ${SLURM_ARRAY_TASK_ID}
#cat iteration.txt
#pwd
cd ..


