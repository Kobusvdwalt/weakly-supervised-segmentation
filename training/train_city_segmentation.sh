#!/bin/bash
#SBATCH --job-name=CitySeg
#SBATCH --output=output_city_segmentation.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch
python3 train_city_segmentation.py