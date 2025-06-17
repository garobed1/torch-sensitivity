#!/bin/tcsh
##### These lines are for Slurm
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -t 2:00:00
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

srun  python3.11 run_torch_samples.py /usr/workspace/bedonian1/torch1d_r1_pilot_V2 /usr/workspace/bedonian1/mean_V2_r6/torch1d_input_r.yml 80000


# [1, 2, 5, 6, 9, 10, 14, 17, 21, 22, 25, 26, 29, 30, 33, 34, 37, 38, 41, 42, 46, 49, 53, 54, 57, 58, 61, 62, 65, 66, 69, 70, 73, 74, 78, 81, 85, 86, 89, 90, 93, 94, 97, 98, 101, 102, 105, 106, 110, 113, 117, 118, 121, 122, 126, 129, 133, 134, 137, 138, 142, 145, 149, 150, 153, 154, 157, 158, 161, 162, 165, 166, 169, 170, 174, 177, 181, 182, 185, 186, 190, 193, 194, 197, 198, 201, 202, 206, 209, 213, 214, 217, 218, 221, 222, 225, 226, 229, 230, 233, 234, 238, 241, 245, 246, 249, 250, 253, 254]