#!/bin/bash
#SBATCH --job-name=voc2012B
#SBATCH --output=output.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch
python3 train.py