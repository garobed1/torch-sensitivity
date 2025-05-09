#!/bin/bash
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -t 9:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -o output.%j              #Output file name
#SBATCH --signal=B:10@300        # Send signal 10 (SIG_USR1) 300s before time limit
##SBATCH -A bedonian1


if [ ! -f "%FINISH_REPLACE" ] ; then
	sbatch --dependency=afterany:$SLURM_JOBID run_tps2d_r1.sh
fi

# creates automatic checkpoints and automatically starts from the most recent checkpoint
%CDIR_REPLACE
srun ~/tps/src/tps --runFile %INPUT_REPLACE
# when finished, replace small timestep infile with large timestep infile

if [ ! -f "%FINISH_REPLACE" ] ; then
    %MV_REPLACE_1
    %MV_REPLACE_2
fi

touch %FINISH_REPLACE
exit 0