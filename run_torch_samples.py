import os
import yaml
from subprocess import run
from mpi4py import MPI

from sample_utils import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


"""
Script to run torch simulations for many samples

Will at some point be able to run sample groups across different models for MLBLUE

torch1d:

python torch1d.py input_file.yml

"""

title = "torch1d-propagation-dev"

home = os.environ["HOME"]
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3_no_4p_to_h"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3_no_4p_to_h_dt"
sample_dir = f"{home}/bedonian1/torch1d_samples_r6" #1e-7 dt, a tenth of the original time

torch1d_exec = f"{home}/torch1d/torch1d.py"
pcomm = 'python3.11'

######### Loop through sample directories in sample_dir
samples = os.listdir(sample_dir)
N = len(samples)

cases = divide_cases(N, size)

for isamp in cases[rank]:

    sample = samples[isamp]

    # TODO: In each sample, detect which models to run
    # For now, just run torch1d based on yml file presence

    dir = sample_dir + '/' + sample 

    run_torch1d = False
    # run_tps, etc.

    # check if there exists a yaml file (and read its contents)
    fstep = 99999999
    for fname in os.listdir(dir):
        if fname.endswith('.yml'):
            run_torch1d = True

            with open(dir + '/' + fname) as f:
                torch1d_in = yaml.safe_load(f)
            
            # get final time step
            fstep = torch1d_in['time_integration']['number_of_timesteps']

            break

    # check if calculation already performed
    if os.path.isdir(dir + '/output/'):
        for fname in os.listdir(dir + '/output/'):
            # if fname.endswith('-00010000.h5'): #completed run NOTE, need to adapt this
            if fname.endswith(f'-{fstep:08d}.h5'): #completed run NOTE, need to adapt this
                run_torch1d = False
                break

    if run_torch1d:
        for fname in os.listdir(dir):
            if fname.endswith('.yml'):
                torch1d_in = fname

        inputfile = dir + '/' + torch1d_in
        
        # run torch1d
        run([pcomm, "torch1d.py", inputfile])

