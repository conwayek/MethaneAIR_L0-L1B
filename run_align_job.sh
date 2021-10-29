#!/bin/bash
 
#SBATCH -t 0-48:00
#SBATCH -N 1
#SBATCH --job-name=align-rf04 
#SBATCH --mem-per-cpu=30000
#SBATCH -o fail0
#SBATCH -e fail0
#SBATCH --mail-type=END
#SBATCH --mail-user=cmiller@fas.harvard.edu
 
# Run simulation
python align.py
 
#EOF
