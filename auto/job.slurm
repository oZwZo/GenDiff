#!/bin/bash
#SBATCH -A gottgens-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=sampler-auto
#SBATCH --output=/rds/user/wz369/hpc-work/GenDiff-sampler-auto/auto/slurm_%j.out
# Usage: sbatch auto/job.slurm <python_script> [args...]
PYBIN=/rds/user/wz369/hpc-work/LIBS/mamba/envs/GenDiff_env/bin/python
cd /rds/user/wz369/hpc-work/GenDiff-sampler-auto
echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES start=$(date)"
$PYBIN "$@"
echo "end=$(date)"
