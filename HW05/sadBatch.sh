#! /bin/bash
#
#PBA -l walltime=00:05:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -W group_list=newriver
#PBS -q p100_dev_q
#PBS -A CMDA3634SP18

cd $PBS_O_WORKDIR

module purge
module load cuda

nvcc -o cudaDecrypt -arch=sm_60 cudaDecrypt.cu

./cudaDecrypt

