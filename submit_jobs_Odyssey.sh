#!/bin/bash

input="ch4_seq_files_RF02.txt"
#input="test.txt"

i=0
while IFS= read -r line

do
echo $i > iteration.txt


mkdir -p "$line"


echo "#!/bin/bash" > job.sh
echo " "  >> job.sh
echo "#SBATCH -t 0-15:00" >> job.sh
echo "#SBATCH -N 1" >> job.sh
echo "#SBATCH --job-name=ch4-n$i " >> job.sh
echo "#SBATCH --mem-per-cpu=50000" >> job.sh
echo "#SBATCH -o fail" >> job.sh
echo "#SBATCH -e fail" >> job.sh
echo "module load R_packages/4.0.5-fasrc01  " >> job.sh
echo "module load udunits/2.2.26-fasrc02  " >> job.sh
echo "module load geos/3.9.1-fasrc01  " >> job.sh
echo "module load proj/8.0.0-fasrc01  " >> job.sh
echo "module load gdal/3.2.2-fasrc01    " >> job.sh
echo "unset R_LIBS_SITE   " >> job.sh
echo "export R_LIBS_USER=/n/home04/econway/apps/R_4.0.2" >> job.sh
echo " " >> job.sh
echo "# Run simulation" >> job.sh
echo "python run_L1B_07_07.py "  >> job.sh
echo " " >> job.sh
echo "#EOF"  >> job.sh

cp wavecal_routines.py dem_maker.py akaze_nc_06_24_2021.py akaze_nc_ch4_o2_06_24_2021.py  job.sh aggregate.py iteration.txt Orthorectification_Avionics_NC_Fast.R Orthorectification_Optimized_NC_Fast_O2.R Orthorectification_Optimized_NC_Fast_CH4.R   MethaneAIR_L1_EKC_07_08_2021.py run_L1B_07_07.py "$line"


cd "$line"

sbatch job.sh

cd ..

i=$(( $i + 1 ))
done < "$input"

