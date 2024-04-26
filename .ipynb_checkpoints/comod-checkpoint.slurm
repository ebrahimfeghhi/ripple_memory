#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --array=1-2%2

echo "Running task number $SLURM_ARRAY_TASK_ID"
python -u /home1/efeghhi/ripple_memory/analysis_code/pac_analyses/comodulogram.py $SLURM_ARRAY_TASK_ID
