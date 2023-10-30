#!/bin/bash
#SBATCH --job-name=matformer_gpu_job
#SBATCH --output=matformer_gpu_job.out
#SBATCH --error=matformer_gpu_job.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus 1
#SBATCH --mail-type=END
#SBATCH --mail-user=s.abbas@deakin.edu.au

module purge

eval "$(conda shell.bash hook)"
conda activate matformer_original
python3 train_folder.py --root_dir /home/abshe/Matformer/my_materials/pristine_materials --config_name /home/abshe/Matformer/my_materials/config.json --output_dir logs/my_materials_pristine --epochs 10000