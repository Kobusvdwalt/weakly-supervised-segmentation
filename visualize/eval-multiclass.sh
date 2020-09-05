#!/bin/bash
#SBATCH --job-name=CAMM
#SBATCH --output=eval-multiclass-output.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch
python3 eval-multiclass.py