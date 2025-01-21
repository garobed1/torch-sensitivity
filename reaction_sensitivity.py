import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py as h5
import scipy.constants as spc

from sample_utils import *
from sobol_tools import *
from pca_tools import *

home = os.getenv('HOME')
sample_dir = home + "/torch-chemistry/argon/results/test_stepwise_2/"

# number of principal components to examine in sensitivity analysis
N_T = 512
N_pc = 2



sample_exc = False
sample_ion = False
sample_step_exc = False
if os.path.exists(sample_dir + 'sig_A_000000/rates/Excitation_res.h5'):
    sample_exc = True
if os.path.exists(sample_dir + 'sig_A_000000/rates/Ionization_res.h5'):
    sample_ion = True
if os.path.exists(sample_dir + 'sig_A_000000/rates/StepExcitation_res_meta.h5'):
    sample_step_exc = True

######## Number of uncertain variables
df = pd.read_csv('~/torch-chemistry/argon/input-data/ArI-levels-nist.csv')
configuration = df['Configuration']
term = df['Term']
J = df['J']

known_configurations = ['4s', '5s', '6s',
                        '4p', '5p', '6p',
                        '3d', '4d', '5d', '6d']

total_config = []
for i in range(0,len(df)):
    cfg = configuration[i] + "-" + term[i] + "-" + str(J[i])
    total_config.append(cfg)

# then keep track of "known" configs only
config_perturb_dist = {}
sizes = []
for i in range(0,len(df)):
    base_config = configuration[i][19:]

    if base_config in known_configurations:

        #NOTE: Only looking at one variable per config for now
        sizes.append(1)
        config_perturb_dist[total_config[i]] = {}

Nconfig = len(list(config_perturb_dist.keys())) 
Nvars_exc = Nconfig
Nvars_ion = Nconfig
Nvars_step_exc = 0
for i in range(Nconfig):
    Nvars_step_exc += i

Nvars = 0
if sample_exc:
    Nvars += Nvars_exc
if sample_ion:
    Nvars += Nvars_ion
if sample_step_exc:
    Nvars += Nvars_step_exc

######## Number of A samples

Nsamples = 0
o_name = sample_dir + "sig_A_{0:06d}/rates".format(Nsamples)
while os.path.isdir(o_name):

    Nsamples += 1
    o_name = sample_dir + "sig_A_{0:06d}/rates".format(Nsamples)

########
# NOTE: We should perform sensitivity analysis on principal components 
lumped_rates = ["meta", "res", "fourp", "higher"]
rate_sizes = [N_T, N_T, N_T, N_T]
pc_sizes = [N_pc, N_pc, N_pc, N_pc]

######### Construct a PCA score model of the inputs, aka KL expansion

# Essentially, use the samples in A (or all?) to estimate \lambda, v principal components
# use all (Lamboni et al. 2011)

# Then, map all A, B, AB samples to PC scores \xi up to some truncation (quantify variance)
# Then perform sobol sensitivity analysis on scores

rate_pc_samples = {"A": SampleData(categories=lumped_rates, sizes=pc_sizes),
                "B": SampleData(categories=lumped_rates, sizes=pc_sizes),
                "AB": SampleData(categories=lumped_rates, sizes=pc_sizes)}

full_rate = {"meta": np.zeros([N_T, (Nvars+2)*Nsamples]),
            "res": np.zeros([N_T, (Nvars+2)*Nsamples]),
            "fourp": np.zeros([N_T, (Nvars+2)*Nsamples]),
            "higher": np.zeros([N_T, (Nvars+2)*Nsamples])}

# Read results back
c_f = 0
for s_data in ["A", "B", "AB"]:

    c = 0
    o_name = sample_dir + "sig_{0}_{1:06d}/rates".format(s_data, c)
    while os.path.isdir(o_name):
        for rate in lumped_rates:
            with h5.File("{0}/Excitation_{1}.h5".format(o_name, rate), "r") as f:
                full_rate[rate][:,c_f] = f["table"][:,1]
        c += 1
        c_f += 1
        o_name = sample_dir + "sig_{0}_{1:06d}/rates".format(s_data, c)

# now compute PCA and assemble scores
mean = {}
eigval = {}
eigvec = {}
scores = {}
for rate in lumped_rates:
    mean[rate], eigval[rate], eigvec[rate], scores[rate] = estimateCovarianceEig(full_rate[rate])


# reassemble scores in SampleData objects
# rate_pc_samples = {"A": SampleData(categories=lumped_rates, sizes=pc_sizes),
#                 "B": SampleData(categories=lumped_rates, sizes=pc_sizes),
#                 "AB": SampleData(categories=lumped_rates, sizes=pc_sizes)}

Ns = {"A":[0, Nsamples], "B":[Nsamples,2*Nsamples], "AB":[2*Nsamples,(Nvars+2)*Nsamples]}
for s_data in ["A", "B", "AB"]:
    dd = {}
    for rate in lumped_rates:
        dd[rate] = scores[rate][:N_pc, Ns[s_data][0]:Ns[s_data][1]]

    # breakpoint()
    rate_pc_samples[s_data].addData(dd)


######### Test Plots

# Constants
qe = spc.e    # 1.60217663e-19 [C]
kB = spc.k    # 1.380649e-23 [J/K]
NA = spc.N_A  # 6.022e23 [#/mol]
eV = qe/kB    # 11604.518 [K / eV]

# temp grid
T_Maxw = np.linspace(0.02, 2, 512) * eV

# check that KL model reproduces well
up_to = [1,2,3,4]
s = 1

y_T = full_rate["meta"][:,s]
y_E = []
for i in up_to:
    y_E.append(mean["meta"] + np.dot(scores["meta"][:i, s], eigvec["meta"][:,:i].T))
y_M = mean["meta"]

plt.plot(T_Maxw, y_M, label="Mean")
plt.plot(T_Maxw, y_T, label=f"Sample {s} (Orig)")
for i in range(len(up_to)):
    plt.plot(T_Maxw, y_E[i], label=f"Sample {s} (KL {up_to[i]})")

plt.xlabel(rf"$T$ [K]")
plt.ylabel(rf"$k_f$ [m$^3$ / mol / s]")
plt.grid()
plt.legend()
plt.savefig(f"argon-excitation-meta-sample-KL.pdf", bbox_inches='tight')

# breakpoint()

######### Estimate Sobol Indices

# compute variance percentage too
frac = {}
r1 = {}
r2 = {}
for rate in lumped_rates:
    print("{0} Indices: ".format(rate))
    frac[rate] = computeVarianceFractions(eigval[rate])
    print(frac[rate])

    r1[rate], r2[rate] = computeSobolIndices(rate_pc_samples['A'], rate_pc_samples['B'], rate_pc_samples['AB'], Nvars, rate)
    print(r1[rate])
    # print(r2[rate])
    print()

    breakpoint()



#TODO TODO TODO TODO TODO TODO
######### Look at Ionization rates
######### Group Indices by Electron Config and 
######### Save useful results to discuss on Wed

