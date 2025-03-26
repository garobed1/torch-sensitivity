import os
import yaml
from subprocess import run
from mpi4py import MPI
import h5py

from sample_utils import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Script to read torch outputs and diagnose issues


"""

home = os.environ["HOME"]
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3_no_4p_to_h"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r3_no_4p_to_h_dt"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r6_1"
# sample_dir = f"{home}/bedonian1/torch1d_samples_r6_3"
sample_dir = f"{home}/bedonian1/torch1d_samples_r7"
# nom_dir = f"{home}/bedonian1/nominal_r6/"
nom_dir = f"{home}/bedonian1/mean_r6/"
# Nsteps_exp = 10000

# fstep_override = None
# fstep_override = 65000
fstep_override = 70000

compare_at = 1023

samples = os.listdir(sample_dir)
Ntotal = len(samples)



cases_completed = [] # cases that reach the final time step
cases_crashed_0 = [] # cases that crash immediately
cases_crashed_1 = [] # cases that crash after a number of time steps
dens_completed = []
dens_crashed_1 = []
cases_notrun = [] # not run yet
cases_stopped = []

# get information from nominal case

# nom_outdir = nom_dir + 'output'
# filename = nom_outdir + '/AxialICPTorch-00000000.h5'
# with h5py.File(filename, 'r') as f:
#     tablenom0 = f['conserved'][...]
# filename = nom_outdir + '/AxialICPTorch-00000100.h5'
# with h5py.File(filename, 'r') as f:
#     tablenom1 = f['conserved'][...]

# offnom_init = (tablenom1[compare_at,-5:] - tablenom0[compare_at,-5:])/tablenom0[compare_at,-5:]

# get sample info
# samples = ['sig_A_000001']
for sample in samples:

    pdir = sample_dir + '/' + sample

    # find input file
    if fstep_override is not None:
        fstep = fstep_override
    else:
        for fname in os.listdir(pdir):
            if fname.endswith('.yml'):
                # run_torch1d = True

                with open(pdir + '/' + fname) as f:
                    torch1d_in = yaml.safe_load(f)
                
                # get final time step
                fstep = torch1d_in['time_integration']['number_of_timesteps']

                break

    outdir = sample_dir + '/' + sample + '/output/' 
    sind = int(sample[-6:])

    was_run = False
    advanced = False
    crashed = False
    completed = False

    if os.path.exists(outdir):
        was_run = True
        
        outfiles = os.listdir(outdir)

        if len(outfiles) > 2:
            advanced = True

        for fname in outfiles:
            # breakpoint()
            if fname.endswith('crashed.h5'):
                crashed = True

            if fname.endswith(f'-{fstep:08d}.h5'):
                completed = True

    if completed:
        crashed = False

    if not was_run:
        # print(f"Case {sind:06d} NOT RUN")
        cases_notrun.append(sind)
    elif not advanced and crashed:
        # print(f"Case {sind:06d} FAILED AT FIRST ITER")
        cases_crashed_0.append(sind)

        # filename = outdir + '/AxialICPTorch-00000000.h5'
        # with h5py.File(filename, 'r') as f:
        #     table0 = f['conserved'][...]

        # filename = outdir + '/AxialICPTorch.crashed.h5'
        # with h5py.File(filename, 'r') as f:
        #     table1 = f['conserved'][...]
        
        # off_init = (table1[compare_at, -5:] - table0[compare_at, -5:])/table0[compare_at,-5:]
        # off_nominal = (table1[compare_at, -5:] - tablenom1[compare_at, -5:])/tablenom1[compare_at,-5:]


        # breakpoint()

    elif advanced and crashed:
        # print(f"Case {sind:06d} FAILED BEFORE FINAL STEP")
        cases_crashed_1.append(sind)

        

    elif completed:
        # print(f"Case {sind:06d} COMPLETED")
        cases_completed.append(sind)

        # filename = outdir + '/AxialICPTorch-00000000.h5'
        # with h5py.File(filename, 'r') as f:
        #     table0 = f['conserved'][...]

        # filename = outdir + '/AxialICPTorch-00000100.h5'
        # with h5py.File(filename, 'r') as f:
        #     table1 = f['conserved'][...]
        
        # off_init = (table1[compare_at, -5:] - table0[compare_at, -5:])/table0[compare_at,-5:]
        # off_nominal = (table1[compare_at, -5:] - tablenom1[compare_at, -5:])/tablenom1[compare_at,-5:]


        # breakpoint()

    else:
        # print(f"Case {sind:06d} RUN BUT MANUALLY STOPPED BEFORE COMPLETE")
        # quit()
        cases_stopped.append(sind)

Ncomplete = len(cases_completed)
Ncrash0 = len(cases_crashed_0)
Ncrash1 = len(cases_crashed_1)
Nnotrun = len(cases_notrun)
Nstopped = len(cases_stopped)

print(f'{Ntotal} Total')
print(f'{Ncomplete} Completed')
print(f'{Ncrash0} Crashed Immediately')
print(f'{Ncrash1} Crashed After Running A Bit')
print(f'{Nnotrun} Not Run')
print(f'{Nstopped} Unknown')



breakpoint()