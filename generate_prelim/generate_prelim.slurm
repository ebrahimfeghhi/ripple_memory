#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --mem=25GB

# Create a list with elements from 1 to 9
brain_regions=(1 2 3 4 5 6 7 8 9)

# Iterate through the list and execute the Python script for each brain region
for region in "${brain_regions[@]}"
do
  echo "Running task number $SLURM_ARRAY_TASK_ID with brain region $region"
  python -u /home1/efeghhi/ripple_memory/create_temp_df.py $region
done