#!/bin/bash
 
#SBATCH -t 0-48:00
#SBATCH -N 1
#SBATCH --job-name=align
#SBATCH --mem-per-cpu=30000
#SBATCH -o fail
#SBATCH -e fail
#SBATCH --mail-type=END
#SBATCH --mail-user=cmiller@fas.harvard.edu
 
# Run simulation
python align.py
 
#EOF
