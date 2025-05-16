#!/bin/bash
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -t 8:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -o output.%j              #Output file name
#SBATCH --signal=B:10@300        # Send signal 10 (SIG_USR1) 300s before time limit
##SBATCH -A bedonian1



# creates automatic checkpoints and automatically starts from the most recent checkpoint
cd ~/bedonian1/mean_tps2d_LF_r6/
srun ~/tps/src/tps --runFile lomach.torch.reacting.ini
# when finished, replace small timestep infile with large timestep infile


exit 0