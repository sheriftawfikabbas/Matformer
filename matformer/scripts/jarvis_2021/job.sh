#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --output=gpu_job.out
#SBATCH --error=gpu_job.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus 1
#SBATCH --mail-type=END
#SBATCH --mail-user=s.abbas@deakin.edu.au

module purge

eval "$(conda shell.bash hook)"
conda activate matformer_original
python3 train.py