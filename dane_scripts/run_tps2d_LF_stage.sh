#!/bin/bash
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -t 3:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -o output.%j              #Output file name
#SBATCH --signal=B:10@300        # Send signal 10 (SIG_USR1) 300s before time limit
##SBATCH -A bedonian1

%CDIR_REPLACE
if [ ! -f "%FINISH_REPLACE" ] ; then
	sbatch --dependency=afterany:$SLURM_JOBID run_tps2d_LF_stage.sh

    # replace successive input files
    %MV_REPLACE_0
    %MV_REPLACE_1
    %MV_REPLACE_2
    %MV_REPLACE_3
    %MV_REPLACE_4
else
    exit 0
fi

# creates automatic checkpoints and automatically starts from the most recent checkpoint
srun ~/tps/src/tps --runFile %INPUT_REPLACE

# if there is no next input file in queue, then we are done
if [ ! -f "%Q_REPLACE" ] ; then
    touch %FINISH_REPLACE
fi

cd -
exit 0