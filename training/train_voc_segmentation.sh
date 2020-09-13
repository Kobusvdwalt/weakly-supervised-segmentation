#!/bin/bash
#SBATCH --job-name=VocSeg
#SBATCH --output=output_voc_segmentation.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch
python3 train_voc_segmentation.py