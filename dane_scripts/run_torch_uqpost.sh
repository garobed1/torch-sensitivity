#!/bin/tcsh
##### These lines are for Slurm
#SBATCH -N 4
#SBATCH -n 448
#SBATCH -t 1:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -o output.%j              #Output file name
###SBATCH -A bedonian1

### Job commands start here 
### Display some diagnostic information
echo '=====================JOB DIAGNOSTICS========================'
date
echo -n 'This machine is ';hostname
echo -n 'My jobid is '; echo $SLURM_JOBID
echo 'My path is:' 
echo $PATH
echo 'My job info:'
squeue -j $SLURM_JOBID
echo 'Machine info'
sinfo -s

echo '=====================JOB STARTING=========================='

srun  python3.11 run_sf_torch1d_uq_post.py