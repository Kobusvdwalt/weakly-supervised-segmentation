#!/bin/bash
#SBATCH --job-name=CAMB
#SBATCH --output=eval-binary-output.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch
python3 eval-binary.py