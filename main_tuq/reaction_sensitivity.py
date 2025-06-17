import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py as h5
import scipy.constants as spc

from util_tuq.sample_utils import *
from util_tuq.sobol_tools import *
from util_tuq.pca_tools import *

"""
Sensitivity analysis

NOTE: Not currently relevant



"""


home = os.getenv('HOME')
sample_dir = home + "/torch-chemistry/argon/results/cross_section_samples_r1/"
res_dir = "../results/cross_section_samples_r1/"

# number of principal components to examine in sensitivity analysis
N_T = 512
N_pc = 4


# NOTE: excluding step exc for now
reaction_types = ['Excitation', 'Deexcitation', 'Ionization', 'Recombination']




sample_exc = False
sample_ion = False
sample_step_exc = False
if os.path.exists(sample_dir + 'sig_A_000000/rates/Excitation_res.h5'):
    sample_exc = True
if os.path.exists(sample_dir + 'sig_A_000000/rates/Ionization_res.h5'):
    sample_ion = True
if os.path.exists(sample_dir + 'sig_A_000000/rates/StepExcitation_res_meta.h5'):
    sample_step_exc = True
if os.path.exists(sample_dir + 'excluded_states.pickle'):
    with open(sample_dir + "excluded_states.pickle", 'rb') as f:
        excluded_states = pickle.load(f)
else:
    excluded_states = []

##########################################################################################################
# Script Starts Here
##########################################################################################################


######## Number of uncertain variables
df = pd.read_csv('~/torch-chemistry/argon/input-data/ArI-levels-nist.csv')
configuration = df['Configuration']
term = df['Term']
J = df['J']
energy_level = df['Level (eV)'].to_numpy()
degeneracy = df['g'].to_numpy()

known_configurations = ['4s', '5s', '6s',
                        '4p', '5p', '6p',
                        '3d', '4d', '5d', '6d']

# get all configs
energy_level_dict, g_dict, total_config = getKnownConfigurations(df, 
                                                                    configuration, 
                                                                    known_configurations, 
                                                                    term, energy_level, 
                                                                    degeneracy, J,
                                                                    excluded_states)


# then keep track of "known" configs only
config_perturb_dist = {}
sizes = []

for cfg in total_config:
    #NOTE: Only looking at one variable per config for now
    sizes.append(1)
    config_perturb_dist[cfg] = {}



config_list = list(config_perturb_dist.keys())

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
lumped_rates_g = ["meta", "res", "fourp", "higher", "Ground"]
rate_sizes = [N_T, N_T, N_T, N_T]
rate_sizes_g = [N_T, N_T, N_T, N_T, N_T]
pc_sizes = [N_pc, N_pc, N_pc, N_pc]
pc_sizes_g = [N_pc, N_pc, N_pc, N_pc, N_pc]

# NOTE: Ground ionization not working at the moment, exclude for now
lumped_rates_g = lumped_rates
rate_sizes_g = rate_sizes
pc_sizes_g = pc_sizes


######### Construct a PCA score model of the inputs, aka KL expansion

# Essentially, use the samples in A (or all?) to estimate \lambda, v principal components
# use all (Lamboni et al. 2011)

# Then, map all A, B, AB samples to PC scores \xi up to some truncation (quantify variance)
# Then perform sobol sensitivity analysis on scores

rate_pc_samples = {}

for rtype in reaction_types:
    if rtype == "Ionization" or rtype == "Recombination":
        rate_pc_samples[rtype] = {"A": SampleData(categories=lumped_rates_g, sizes=pc_sizes_g),
                "B": SampleData(categories=lumped_rates_g, sizes=pc_sizes_g),
                "AB": SampleData(categories=lumped_rates_g, sizes=pc_sizes_g)}
    else:
        rate_pc_samples[rtype] = {"A": SampleData(categories=lumped_rates, sizes=pc_sizes),
                "B": SampleData(categories=lumped_rates, sizes=pc_sizes),
                "AB": SampleData(categories=lumped_rates, sizes=pc_sizes)}

full_rate = {}
for rtype in reaction_types:
    full_rate[rtype] = {"meta": np.zeros([N_T, (Nvars+2)*Nsamples]),
            "res": np.zeros([N_T, (Nvars+2)*Nsamples]),
            "fourp": np.zeros([N_T, (Nvars+2)*Nsamples]),
            "higher": np.zeros([N_T, (Nvars+2)*Nsamples])}

if "Ionization" in reaction_types:
    full_rate["Ionization"]["Ground"] = np.zeros([N_T, (Nvars+2)*Nsamples])
if "Recombination" in reaction_types:
    full_rate["Recombination"]["Ground"] = np.zeros([N_T, (Nvars+2)*Nsamples])

# Read results back
c_f = 0
for s_data in ["A", "B", "AB"]:

    c = 0
    o_name = sample_dir + "sig_{0}_{1:06d}/rates".format(s_data, c)
    while os.path.isdir(o_name):
        for rate in lumped_rates:
            for rtype in reaction_types:
                with h5.File("{0}/{1}_{2}.h5".format(o_name, rtype, rate), "r") as f:
                    full_rate[rtype][rate][:,c_f] = f["table"][:,1]

                # with h5.File("{0}/Deexcitation_{1}.h5".format(o_name, rate), "r") as f:
                #     full_rate_deexc[rate][:,c_f] = f["table"][:,1]

                # with h5.File("{0}/Ionization_{1}.h5".format(o_name, rate), "r") as f:
                #     full_rate_ion[rate][:,c_f] = f["table"][:,1]

                # with h5.File("{0}/Recombination_{1}.h5".format(o_name, rate), "r") as f:
                #     full_rate_recom[rate][:,c_f] = f["table"][:,1]

        if "Ionization" in reaction_types:
            with h5.File("{0}/Recombination_Ground.h5".format(o_name), "r") as f:
                full_rate["Ionization"]["Ground"][:,c_f] = f["table"][:,1]

        if "Recombination" in reaction_types:
            with h5.File("{0}/Recombination_Ground.h5".format(o_name), "r") as f:
                full_rate["Recombination"]["Ground"][:,c_f] = f["table"][:,1]

        c += 1
        c_f += 1
        o_name = sample_dir + "sig_{0}_{1:06d}/rates".format(s_data, c)

# now compute PCA and assemble scores
mean = {}
eigval = {}
eigvec = {}
scores = {}

for rtype in reaction_types:
    mean[rtype] = {}
    eigval[rtype] = {}
    eigvec[rtype] = {}
    scores[rtype] = {}

    for rate in lumped_rates:
        mean[rtype][rate], eigval[rtype][rate], eigvec[rtype][rate], scores[rtype][rate] = estimateCovarianceEig(full_rate[rtype][rate])


    if rtype == "Ionization" or rtype == "Recombination":
        try:
            mean[rtype]["Ground"], eigval[rtype]["Ground"], eigvec[rtype]["Ground"], scores[rtype]["Ground"] = estimateCovarianceEig(full_rate[rtype]["Ground"])
        except:
            mean[rtype]["Ground"], eigval[rtype]["Ground"], eigvec[rtype]["Ground"], scores[rtype]["Ground"] = None, None, None, None

# reassemble scores in SampleData objects
# rate_pc_samples = {"A": SampleData(categories=lumped_rates, sizes=pc_sizes),
#                 "B": SampleData(categories=lumped_rates, sizes=pc_sizes),
#                 "AB": SampleData(categories=lumped_rates, sizes=pc_sizes)}

Ns = {"A":[0, Nsamples], "B":[Nsamples,2*Nsamples], "AB":[2*Nsamples,(Nvars+2)*Nsamples]}

for rtype in reaction_types:
    for s_data in ["A", "B", "AB"]:
        dd = {}

        if rtype == "Ionization" or rtype == "Recombination":            
            for rate in lumped_rates_g:
                dd[rate] = scores[rtype][rate][:N_pc, Ns[s_data][0]:Ns[s_data][1]]
        else:
            for rate in lumped_rates:
                dd[rate] = scores[rtype][rate][:N_pc, Ns[s_data][0]:Ns[s_data][1]]

        # breakpoint()
        rate_pc_samples[rtype][s_data].addData(dd)


######### Test Plots

# Constants
qe = spc.e    # 1.60217663e-19 [C]
kB = spc.k    # 1.380649e-23 [J/K]
NA = spc.N_A  # 6.022e23 [#/mol]
eV = qe/kB    # 11604.518 [K / eV]

# temp grid
T_Maxw = np.linspace(0.02, 2, 512) * eV

ptype = "Recombination"
prate = "meta"

# check that KL model reproduces well
up_to = [1,2,3,4]
s = 10

y_T = full_rate[ptype][prate][:,s]
y_E = []
for i in up_to:
    y_E.append(mean[ptype][prate] + np.dot(scores[ptype][prate][:i, s], eigvec[ptype][prate][:,:i].T))
y_M = mean[ptype][prate]

plt.plot(T_Maxw, y_M, label="Mean")
plt.plot(T_Maxw, y_T, label=f"Sample {s} (Orig)")
for i in range(len(up_to)):
    plt.plot(T_Maxw, y_E[i], label=f"Sample {s} (KL {up_to[i]})")

plt.xlabel(rf"$T$ [K]")
plt.ylabel(rf"$k_f$ [m$^3$ / mol / s]")
plt.grid()
plt.legend()
plt.savefig(res_dir + f"plots/argon-{ptype}-{prate}-sample-{s}-KL.pdf", bbox_inches='tight')

# breakpoint()

######### Estimate Sobol Indices

# compute variance percentage too
frac = {}
r1 = {}
r2 = {}

for rtype in reaction_types:
    frac[rtype] = {}
    r1[rtype] = {}
    r2[rtype] = {}

    lumped_rates_check = lumped_rates
    if rtype == "Ionization" or rtype == "Recombination":  
        lumped_rates_check = lumped_rates_g

    for rate in lumped_rates_check:
        print("{0} Indices: ".format(rate))
        frac[rtype][rate] = computeVarianceFractions(eigval[rtype][rate])
        print(frac[rtype][rate])

        r1[rtype][rate], r2[rtype][rate] = computeSobolIndices(rate_pc_samples[rtype]['A'], rate_pc_samples[rtype]['B'], rate_pc_samples[rtype]['AB'], Nvars, rate)
        print(r1[rtype][rate])
        # print(r2[rate])
        print()
        # breakpoint()



with open(res_dir + "/mean.pickle", 'wb') as f:
    pickle.dump(mean ,f)
with open(res_dir + "/eigval.pickle", 'wb') as f:
    pickle.dump(eigval ,f)
with open(res_dir + "/eigvec.pickle", 'wb') as f:
    pickle.dump(eigvec ,f)
with open(res_dir + "/scores.pickle", 'wb') as f:
    pickle.dump(scores ,f)
with open(res_dir + "/frac.pickle", 'wb') as f:
    pickle.dump(scores ,f)
with open(res_dir + "/r1.pickle", 'wb') as f:
    pickle.dump(r1 ,f)
with open(res_dir + "/r2.pickle", 'wb') as f:
    pickle.dump(r2 ,f)


# breakpoint()



#TODO TODO TODO TODO TODO TODO
######### Look at Ionization rates - Done
######### Group Indices by Electron Config and 
######### Save useful results to discuss on Wed

