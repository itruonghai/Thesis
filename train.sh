#!/bin/bash
#SBATCH --job-name=thesis            # create a short name for your job
#SBATCH --output=/home/nthai/SegTransVAE/slurm2.out      # create a output file
#SBATCH --error=/home/nthai/SegTransVAE/slurm2.err       # create a error file
#SBATCH --partition=batch          # choose partition
#SBATCH --gres=gpu:2                # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --cpus-per-task=8          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=3-0:00:00

echo   Date              = $(date)
echo   Hostname          = $(hostname -s)
echo   Working Directory = $(pwd)
echo   Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES
echo   Number of Tasks Allocated      = $SLURM_NTASKS
echo   Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK

# Reconfigure HOME_PATH
# pwd

cd /home/nthai/
source thesis/bin/activate

cd /home/nthai/SegTransVAE

python lightning_train.py --exp segformer_new
# python test.py
