#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --mem=45GB
#SBATCH --array=2-3%2

echo "Running task number $SLURM_ARRAY_TASK_ID"
python -u /home1/efeghhi/ripple_memory/analysis_code/pac_analyses/call_theta_gamma_PAC.py $SLURM_ARRAY_TASK_ID
