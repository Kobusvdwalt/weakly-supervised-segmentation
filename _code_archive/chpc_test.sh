#!/bin/bash
#PBS -N gpu_test
#PBS -q gpu_1
#PBS -l ncpus=8:ngpus=1
#PBS -P CSCI1340
#PBS -l walltime=4:00:00
#PBS -o /mnt/lustre/users/pvanderwalt1/cuda_test/test1.out
#PBS -e /mnt/lustre/users/pvanderwalt1/cuda_test/test1.err
#PBS -m abe
#PBS -M kobusvdwalt9@gmail.com
 
cd /mnt/lustre/users/pvanderwalt1/weakly-supervised-segmentation/training/
 
echo
echo `date`: executing CUDA job on host ${HOSTNAME}
echo
 
# Run program
python train_voc_segmentation.py