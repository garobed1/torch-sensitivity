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
# from utils.rate_utils import *

home = os.getenv('HOME')
# sample_dir = home + "/torch-chemistry/argon/results/cross_section_samples_r3/"
# sample_dir = home + "/bedonian1/cross_section_samples_r6/"
sample_dir = home + "/bedonian1/cross_section_samples_r7/"
nom_dir = home + "/torch-sensitivity/trevilo-cases/torch_7sp_chem/nominal/rate-coefficients/"
# res_dir = "results/cross_section_samples_r3/"
# res_dir = "results/cross_section_samples_r6_3/"
res_dir = "results/cross_section_samples_r7/"

mean_dir = home + "/bedonian1/mean_r6/"

"""
Without performing sensitivity analysis, simply build KL expansions of the rate 
samples

NOTE NOTE NOTE: Check notes to see where I left off

"""

# number of principal components to examine in sensitivity analysis
N_T = 512
N_pc = 6

make_plots = False
# number of samples to plot
Ndraw = 1000
clim = 200

# NOTE: excluding step exc for now
reaction_types = ['Excitation', 'Deexcitation', 'Ionization', 'Recombination']
reaction_types_full = ['Excitation', 'Deexcitation', 'Ionization', 'Recombination', 'StepExcitation']

# seemingly crash no matter what
# crashed_runs =[30, 76, 232, 370, 416, 444, 453, 526, 535, 592, 618, 703, 839, 1217, 1229, 1234, 1302, 1373, 1569, 1721, 1857, 2032, 2254, 2466, 2805]
crashed_runs = []

# Constants
qe = spc.e    # 1.60217663e-19 [C]
kB = spc.k    # 1.380649e-23 [J/K]
NA = spc.N_A  # 6.022e23 [#/mol]
eV = qe/kB    # 11604.518 [K / eV]
# temp grid
T_Maxw = np.linspace(0.02, 2, 512) * eV


# sample_labels = ["A", "B", "AB"]
sample_labels = ["A"]

######## Script begins here

# if rank == 0:
if not os.path.isdir(res_dir):
    os.makedirs(res_dir)

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

######## Number of samples

Nsamples = 0
o_name = sample_dir + "sig_A_{0:06d}/rates".format(Nsamples)
while os.path.isdir(o_name):

    Nsamples += 1
    o_name = sample_dir + "sig_A_{0:06d}/rates".format(Nsamples)

########
lumped_rates = ["meta", "res", "fourp", "higher"]
lumped_rates_g = ["meta", "res", "fourp", "higher", "Ground"]

rate_sizes = {}
pc_sizes = {}
rate_sizes_g = {}
pc_sizes_g = {}

for key in lumped_rates:

    rate_sizes[key] = N_T
    pc_sizes[key] = N_pc

for key in lumped_rates_g:

    rate_sizes_g[key] = N_T
    pc_sizes_g[key] = N_pc


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
    rate_pc_samples[rtype] = {}
    for label in sample_labels:
        if rtype == "Ionization" or rtype == "Recombination":
            rate_pc_samples[rtype][label] = SampleData(categories=lumped_rates_g, sizes=pc_sizes_g)
        else:
            rate_pc_samples[rtype][label] = SampleData(categories=lumped_rates, sizes=pc_sizes)

# load rate samples nominal rates
nom_rate = {}
full_rate = {}
process_label = {}
unit_label = {}
for rtype in reaction_types:
    full_rate[rtype] = {"meta": np.zeros([N_T, Nsamples]),
            "res": np.zeros([N_T, Nsamples]),
            "fourp": np.zeros([N_T, Nsamples]),
            "higher": np.zeros([N_T, Nsamples])}
    
    nom_rate[rtype] = {"meta": np.zeros([N_T, 1]),
            "res": np.zeros([N_T, 1]),
            "fourp": np.zeros([N_T, 1]),
            "higher": np.zeros([N_T, 1])}
    
    process_label[rtype] = {}
    unit_label[rtype] = {}
    
full_rate['StepExcitation'] = {}
nom_rate['StepExcitation'] = {}
process_label['StepExcitation'] = {}
unit_label['StepExcitation'] = {}
for rate_i in lumped_rates:
    for rate_j in lumped_rates:
        if rate_i == rate_j:
            continue

        rname = f'{rate_i}_{rate_j}'

        full_rate['StepExcitation'][rname] = np.zeros([N_T, Nsamples])
        nom_rate['StepExcitation'][rname] = np.zeros([N_T, 1])
# breakpoint()

if "Ionization" in reaction_types:
    full_rate["Ionization"]["Ground"] = np.zeros([N_T, Nsamples])
    nom_rate["Ionization"]["Ground"] = np.zeros([N_T, 1])
if "Recombination" in reaction_types:
    full_rate["Recombination"]["Ground"] = np.zeros([N_T, Nsamples])
    nom_rate["Recombination"]["Ground"] = np.zeros([N_T, 1])





# Read nominal rates
for rate in lumped_rates:
    for rtype in reaction_types:
        with h5.File("{0}/{1}_{2}.h5".format(nom_dir, rtype, rate), "r") as f:
            nom_rate[rtype][rate][:,0] = f["table"][:,1]
            process_label[rtype][rate] = f.attrs["process"]
            unit_label[rtype][rate] = f.attrs["rate units"]

if "Ionization" in reaction_types:
    with h5.File("{0}/Ionization_Ground.h5".format(nom_dir), "r") as f:
        nom_rate["Ionization"]["Ground"][:,0] = f["table"][:,1]
        process_label["Ionization"]["Ground"] = f.attrs["process"]
        unit_label["Ionization"]["Ground"] = f.attrs["rate units"]

if "Recombination" in reaction_types:
    with h5.File("{0}/Recombination_Ground.h5".format(nom_dir), "r") as f:
        nom_rate["Recombination"]["Ground"][:,0] = f["table"][:,1]

# override
process_label["Recombination"]["Ground"] = 'Ar+ + e + e -> Ar(Ground) + e'
unit_label["Recombination"]["Ground"] = 'm^6 / mol^2 / s'

# Step Excitation
for rate_i in lumped_rates:
    for rate_j in lumped_rates:
        if rate_i == rate_j:
            continue

        rname = f'{rate_i}_{rate_j}'

        with h5.File("{0}/{1}_{2}_{3}.h5".format(nom_dir, 'StepExcitation', rate_i, rate_j), "r") as f:
            nom_rate['StepExcitation'][rname][:,0] = f["table"][:,1]
            process_label['StepExcitation'][rname] = f.attrs["process"]
            unit_label['StepExcitation'][rname] = f.attrs["rate units"]
# breakpoint()
# Read results back
c = {}

c_f = 0
for s_data in sample_labels:

    c[s_data] = 0
    o_name = sample_dir + "sig_{0}_{1:06d}/rates".format(s_data, c[s_data])
    while os.path.isdir(o_name) or c[s_data] < clim:
        for rate in lumped_rates:
            for rtype in reaction_types:
                with h5.File("{0}/{1}_{2}.h5".format(o_name, rtype, rate), "r") as f:
                    full_rate[rtype][rate][:,c_f] = f["table"][:,1]
                    


        if "Ionization" in reaction_types:
            with h5.File("{0}/Ionization_Ground.h5".format(o_name), "r") as f:
                full_rate["Ionization"]["Ground"][:,c_f] = f["table"][:,1]           

        if "Recombination" in reaction_types:
            with h5.File("{0}/Recombination_Ground.h5".format(o_name), "r") as f:
                full_rate["Recombination"]["Ground"][:,c_f] = f["table"][:,1]

        # Step Excitation
        for rate_i in lumped_rates:
            for rate_j in lumped_rates:
                if rate_i == rate_j:
                    continue

                rname = f'{rate_i}_{rate_j}'

                with h5.File("{0}/{1}_{2}_{3}.h5".format(o_name, 'StepExcitation', rate_i, rate_j), "r") as f:
                    full_rate['StepExcitation'][rname][:,c_f] = f["table"][:,1]
                    
                    # if rate_i == 'fourp' and rate_j == 'higher':
                        # breakpoint()
        c[s_data] += 1
        c_f += 1
        o_name = sample_dir + "sig_{0}_{1:06d}/rates".format(s_data, c[s_data])




# breakpoint()

# rate sample plots

if make_plots:

    upto = -1
    # upto = 50
    for s_data in sample_labels:

        for ptype in reaction_types_full:
            for rname in full_rate[ptype].keys():
                
                # if ptype == 'Ionization' and rname == 'meta':
                #     breakpoint()
                # ptype = "Excitation"
                # prate = "meta"

                Nind = np.random.choice(c[s_data], Ndraw,  replace=False)

                y_T = full_rate[ptype][rname][:,Nind][:upto,:]
                y_B = full_rate[ptype][rname][:,crashed_runs][:upto,:]
                y_N = nom_rate[ptype][rname][:,0][:upto]
                # y_E = []
                # for i in up_to:
                #     y_E.append(mean[ptype][prate] + np.dot(scores[ptype][prate][:i, s], eigvec[ptype][prate][:,:i].T))
                # y_M = mean[ptype][prate]

                plt.plot([], [], label = "Samples",  color='r', alpha = 0.5)
                plt.plot(T_Maxw[:upto], y_T, color='r', alpha = 0.5)
                plt.plot([], [], label = "Crashed",  color='b', alpha = 1.0)
                plt.plot(T_Maxw[:upto], y_B, color='b', alpha = 1.0)
                plt.plot(T_Maxw[:upto], y_N, label="Nominal")
                # for i in range(len(up_to)):
                #     plt.plot(T_Maxw, y_E[i], label=f"Sample {s} (KL {up_to[i]})")

                plt.xlabel(rf"$T$ [K]")
                plt.ylabel(rf"$k_f$ [m$^3$ / mol / s]")
                plt.yscale('log')
                plt.grid()
                plt.legend()
                plt.savefig(res_dir + f"plots/argon-{ptype}-{rname}-samples.pdf", bbox_inches='tight')
                plt.clf()

    breakpoint()


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

    # breakpoint()
    for rate in lumped_rates:
        mean[rtype][rate], eigval[rtype][rate], eigvec[rtype][rate], scores[rtype][rate] = estimateCovarianceEig(full_rate[rtype][rate])


    if rtype == "Ionization" or rtype == "Recombination":
        try:
            mean[rtype]["Ground"], eigval[rtype]["Ground"], eigvec[rtype]["Ground"], scores[rtype]["Ground"] = estimateCovarianceEig(full_rate[rtype]["Ground"])
        except:
            mean[rtype]["Ground"], eigval[rtype]["Ground"], eigvec[rtype]["Ground"], scores[rtype]["Ground"] = None, None, None, None

# stepwise excitation
rtype = "StepExcitation"
mean[rtype] = {}
eigval[rtype] = {}
eigvec[rtype] = {}
scores[rtype] = {}
for rate_i in lumped_rates:
    for rate_j in lumped_rates:
        if rate_i == rate_j:
            continue

        rate = f'{rate_i}_{rate_j}'

        mean[rtype][rate], eigval[rtype][rate], eigvec[rtype][rate], scores[rtype][rate] = estimateCovarianceEig(full_rate[rtype][rate])# breakpoint()

# write the mean rates to a directory for later use

for rtype in reaction_types_full:
    for rate in mean[rtype].keys():
        table = np.zeros((T_Maxw.shape[0], 2))
        fname = os.path.join(mean_dir,  rtype + '_' + rate + '.h5')
        table[:,0] = T_Maxw[:]
        table[:,1] = mean[rtype][rate]
        with h5.File(fname, 'w') as f:
            f.attrs['process'] = process_label[rtype][rate]
            f.attrs['temperature units'] = 'K'
            f.attrs['rate units'] = unit_label[rtype][rate]
            dset = f.create_dataset("table", table.shape, data=table)




# reassemble scores in SampleData objects
# rate_pc_samples = {"A": SampleData(categories=lumped_rates, sizes=pc_sizes),
#                 "B": SampleData(categories=lumped_rates, sizes=pc_sizes),
#                 "AB": SampleData(categories=lumped_rates, sizes=pc_sizes)}

# Ns = {"A":[0, Nsamples], "B":[Nsamples,2*Nsamples], "AB":[2*Nsamples, Nsamples]}

# for s_data in sample_labels:
#     for rtype in reaction_types:
    
#         dd = {}

#         if rtype == "Ionization" or rtype == "Recombination":            
#             for rate in lumped_rates_g:
#                 dd[rate] = scores[rtype][rate][:N_pc, Ns[s_data][0]:Ns[s_data][1]]
#         else:
#             for rate in lumped_rates:
#                 dd[rate] = scores[rtype][rate][:N_pc, Ns[s_data][0]:Ns[s_data][1]]

#         # breakpoint()
#         rate_pc_samples[rtype][s_data].addData(dd)


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
    pickle.dump(frac ,f)
with open(res_dir + "/r1.pickle", 'wb') as f:
    pickle.dump(r1 ,f)
with open(res_dir + "/r2.pickle", 'wb') as f:
    pickle.dump(r2 ,f)


# breakpoint()





