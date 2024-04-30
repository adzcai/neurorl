#!/bin/bash
#SBATCH --job-name=chaining        # Job name
#SBATCH --output=chaining.out           # Standard output file
#SBATCH --error=chaining.err             # Standard error file
#SBATCH --partition=sapphire    # Partition or queue name
#SBATCH --mem=400GB
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --time=36:00:00                # Maximum runtime (D-HH:MM:SS)

#Load necessary modules (if needed)
module load python/3.10.12-fasrc01
mamba activate neurorl

#Your job commands go here
cd /n/home04/yichenli/human-sf
python envs/blocksworld/AC/experiments.py >> chaininglog.txt
