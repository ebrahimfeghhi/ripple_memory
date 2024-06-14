#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --array=0-1%2

echo "Running task number $SLURM_ARRAY_TASK_ID"
python -u /home1/efeghhi/ripple_memory/analysis_code/pac_analyses/run_comod.py $SLURM_ARRAY_TASK_ID
