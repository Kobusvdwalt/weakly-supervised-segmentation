#!/bin/bash
#SBATCH --job-name=VocClas
#SBATCH --output=output_voc_classification.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch
python3 train_voc_classification.py