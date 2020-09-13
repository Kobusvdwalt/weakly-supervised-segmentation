#!/bin/bash
#SBATCH --job-name=CC
#SBATCH --output=output_city_classification.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch
python3 train_city_classification.py