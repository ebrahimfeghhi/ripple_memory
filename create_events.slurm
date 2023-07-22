#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --array=0-283%3
#SBATCH --time=12:00:00

echo "Running task number $SLURM_ARRAY_TASK_ID"

#python /home1/efeghhi/ripple_memory/play_func.py $SLURM_ARRAY_TASK_ID
python -u /home1/efeghhi/ripple_memory/create_events.py $SLURM_ARRAY_TASK_ID
