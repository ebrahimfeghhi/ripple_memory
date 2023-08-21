#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --array=0-610%12

echo "Running task number $SLURM_ARRAY_TASK_ID"
python -u /home1/efeghhi/ripple_memory/create_events_mne.py $SLURM_ARRAY_TASK_ID