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
# sample_dirs = [home + "/bedonian1/cross_section_samples_r7/"]
# sample_dirs = [home + "/bedonian1/cross_section_samples_r7/", home + "/bedonian1/cross_section_samples_r7_1/"]
# sample_dirs = [home + "/bedonian1/cross_section_samples_r8_1/", home + "/bedonian1/cross_section_samples_r8_2/"]
sample_dirs = [home + "/bedonian1/cross_section_4species_r8_1/", home + "/bedonian1/cross_section_4species_r8_2/"]
nom_dir = home + "/bedonian1/mean_4s_r6/"
# res_dir = "results/cross_section_samples_r3/"
# res_dir = "results/cross_section_samples_r6_3/"
# res_dir = "results/rate_resample_r7/"
# res_dir = "results/rate_resample_r8/"
# res_dir = home + "/bedonian1/rate_resample_model_r8/"
res_dir = home + "/bedonian1/rate_resample_model_4s_r8/"

mean_dir = home + "/bedonian1/mean_4s_r6/"

log_model = True
indep_approach = False

# default is seven species with step excitation
# four_species = False
four_species = True
"""
Without performing sensitivity analysis, simply build KL expansions of the rate 
samples

Since forward and backward rates are correlated, we take a multiple uncorrelated KL
(muKL) approach to obtain simultaneous representations for both with a single sample
of KL parameter scores

Cite H Cho (2013) for muKL approach

NOTE: For frac, define as spectrum eigval_k * ||eigvec^{f or b}_k ||^2_2
Check fraction for forward and backward separately, e.g.
"""

# number of principal components to examine in sensitivity analysis
N_T = 512
N_pc = 6

make_plots = False
# number of samples to plot
Ndraw = 500
clim = 20000

# NOTE: excluding step exc for now
reaction_types = ['Excitation', 'Deexcitation', 'Ionization', 'Recombination']
reaction_types_full = ['Excitation', 'Deexcitation', 'Ionization', 'Recombination', 'StepExcitation']

# seemingly crash no matter what
# crashed_runs =[30, 76, 232, 370, 416, 444, 453, 526, 535, 592, 618, 703, 839, 1217, 1229, 1234, 1302, 1373, 1569, 1721, 1857, 2032, 2254, 2466, 2805]
crashed_runs = []


plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Palatino"],
    "font.size": 16,
})

# Pairings of forward and backward lumped rates
reaction_pairs = [
    ['Excitation', 'Deexcitation'],
    ['Ionization', 'Recombination']
    # ['StepExcitation', 'StepExcitation']
]

# Constants
qe = spc.e    # 1.60217663e-19 [C]
kB = spc.k    # 1.380649e-23 [J/K]
NA = spc.N_A  # 6.022e23 [#/mol]
eV = qe/kB    # 11604.518 [K / eV]
# temp grid
T_Maxw = np.linspace(0.02, 2, N_T) * eV


# sample_labels = ["A", "B", "AB"]
sample_labels = ["A"]

######## Script begins here

# if rank == 0:
if not os.path.isdir(res_dir):
    os.makedirs(res_dir)

sample_exc = False
sample_ion = False
sample_step_exc = False
sample_dir = sample_dirs[0]
if os.path.exists(sample_dir + 'sig_A_000000/rates/Excitation_res.h5') or os.path.exists(sample_dir + 'sig_A_000000/rates/Excitation_Lumped.h5'):
    sample_exc = True
if os.path.exists(sample_dir + 'sig_A_000000/rates/Ionization_Ground.h5'):
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

for sample_dir in sample_dirs:
    Nsf = 0
    o_name = sample_dir + "sig_A_{0:06d}/rates".format(Nsf)
    while os.path.isdir(o_name):

        Nsamples += 1
        Nsf += 1
        o_name = sample_dir + "sig_A_{0:06d}/rates".format(Nsf)

# breakpoint()
########
if four_species:
    lumped_rates = ["Lumped"]
    lumped_rates_g = ["Lumped", "Ground"]
else:
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

    full_rate[rtype] = {}
    for rate in lumped_rates:
        full_rate[rtype][rate] = np.zeros([N_T, Nsamples])
        # full_rate[rtype] = {"meta": np.zeros([N_T, Nsamples]),
        #     "res": np.zeros([N_T, Nsamples]),
        #     "fourp": np.zeros([N_T, Nsamples]),
        #     "higher": np.zeros([N_T, Nsamples])}
    
    nom_rate[rtype] = {}
    for rate in lumped_rates:
        nom_rate[rtype][rate] = np.zeros([N_T, 1])
        # nom_rate[rtype] = {"meta": np.zeros([N_T, 1]),
        #     "res": np.zeros([N_T, 1]),
        #     "fourp": np.zeros([N_T, 1]),
        #     "higher": np.zeros([N_T, 1])}
    
    process_label[rtype] = {}
    unit_label[rtype] = {}
    
if sample_step_exc:
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
if sample_step_exc:
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
if not os.path.exists(res_dir + "/full_rate.pickle"):
    c_f = 0
    for sample_dir in sample_dirs:
        for s_data in sample_labels:
            c[s_data] = 0
            o_name = sample_dir + "sig_{0}_{1:06d}/rates".format(s_data, c[s_data])
            # while os.path.isdir(o_name) or c[s_data] < clim:
            while os.path.isdir(o_name) and c[s_data] < clim:
                print(f"Reading Sample: {c_f}")
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

                if sample_step_exc:
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



    with open(res_dir + "/full_rate.pickle", 'wb') as f:
        pickle.dump(full_rate ,f)

else:
    with open(res_dir + "/full_rate.pickle", 'rb') as f:
        full_rate = pickle.load(f)
    for s_data in sample_labels:
        c[s_data] = min(full_rate['Excitation']['meta'].shape[1], clim)

# breakpoint()

# rate sample plots

col_dict = {'meta':'r',
            'res':'b',
            'fourp':'g',
            'higher':'y',
            'Lumped':'r'
}

label_dict = {'meta':'meta',
            'res':'res',
            'fourp':'4p',
            'higher':'higher',
            'Lumped':'lumped'
}

if make_plots:

    upto = -1
    doto = 0
    # doto = 300
    # upto = 50
    for s_data in sample_labels:

        for ptype in reaction_types_full:
        # for ptype in ['Excitation']:
            for rname in full_rate[ptype].keys():
                
                # if ptype == 'Ionization' and rname == 'meta':
                #     breakpoint()
                # ptype = "Excitation"
                # prate = "meta"

                Nind = np.random.choice(c[s_data], Ndraw,  replace=False)
                # breakpoint()
                y_T = full_rate[ptype][rname][:,Nind][doto:upto,:]
                y_B = full_rate[ptype][rname][:,crashed_runs][doto:upto,:]
                y_N = nom_rate[ptype][rname][:,0][doto:upto]
                # y_E = []
                # for i in up_to:
                #     y_E.append(mean[ptype][prate] + np.dot(scores[ptype][prate][:i, s], eigvec[ptype][prate][:,:i].T))
                # y_M = mean[ptype][prate]

                plt.plot([], [], label = "Samples",  color='r', alpha = 0.5)
                plt.plot(T_Maxw[doto:upto], y_T, color='r', alpha = 0.5)
                # plt.plot([], [], label = f"Lumped {label_dict[rname]}",  color=col_dict[rname], alpha = 1.0)
                # plt.plot(T_Maxw[doto:upto], y_T, color=col_dict[rname], alpha = 0.1)
                plt.plot([], [], label = "Crashed",  color='b', alpha = 1.0)
                plt.plot(T_Maxw[doto:upto], y_B, color='b', alpha = 1.0)
                plt.plot(T_Maxw[doto:upto], y_N, label="Nominal")
                # for i in range(len(up_to)):
                #     plt.plot(T_Maxw, y_E[i], label=f"Sample {s} (KL {up_to[i]})")

                plt.xlabel(rf"$T$ [K]")
                plt.ylabel(rf"$k_f$ [m$^3$ / mol / s]")
                plt.yscale('log')
                plt.grid()
                # plt.legend()
                plt.savefig(res_dir + f"plots/argon-{ptype}-{rname}-samples.pdf", bbox_inches='tight')
                plt.clf()

            # plt.xlabel(rf"$T$ [K]")
            # plt.ylabel(rf"$k_f$ [m$^3$ / mol / s]")
            # plt.yscale('log')
            # plt.ylim([1e-7, 1e10])
            # plt.grid()
            # plt.legend()
            # plt.savefig(res_dir + f"plots/argon-{ptype}-samples.png", dpi=600, bbox_inches='tight')
            # plt.clf()
            # breakpoint()


    quit()

# now compute PCA and assemble scores
mean = {}
eigval = {}
eigvec = {}
scores = {}

if indep_approach:


    for rtype in reaction_types:
        mean[rtype] = {}
        eigval[rtype] = {}
        eigvec[rtype] = {}
        scores[rtype] = {}

        # breakpoint()
        for rate in lumped_rates:

            work = full_rate[rtype][rate]
            if log_model:
                work = np.log(full_rate[rtype][rate])

            mean[rtype][rate], eigval[rtype][rate], eigvec[rtype][rate], scores[rtype][rate] = estimateCovarianceEig(work)


        if rtype == "Ionization" or rtype == "Recombination":
            # try:
            work = full_rate[rtype]["Ground"]
            if log_model:
                work = np.log(full_rate[rtype]["Ground"])
            mean[rtype]["Ground"], eigval[rtype]["Ground"], eigvec[rtype]["Ground"], scores[rtype]["Ground"] = estimateCovarianceEig(work)
            # except:
            #     mean[rtype]["Ground"], eigval[rtype]["Ground"], eigvec[rtype]["Ground"], scores[rtype]["Ground"] = None, None, None, None

    if sample_step_exc:
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

                work = full_rate[rtype][rate]
                if log_model:
                    work = np.log(full_rate[rtype][rate])

                mean[rtype][rate], eigval[rtype][rate], eigvec[rtype][rate], scores[rtype][rate] = estimateCovarianceEig(work)# breakpoint()

    # write the mean rates to a directory for later use

    # for rtype in reaction_types_full:
    #     for rate in mean[rtype].keys():
    #         table = np.zeros((T_Maxw.shape[0], 2))
    #         fname = os.path.join(mean_dir,  rtype + '_' + rate + '.h5')
    #         table[:,0] = T_Maxw[:]
    #         table[:,1] = mean[rtype][rate]
    #         with h5.File(fname, 'w') as f:
    #             f.attrs['process'] = process_label[rtype][rate]
    #             f.attrs['temperature units'] = 'K'
    #             f.attrs['rate units'] = unit_label[rtype][rate]
    #             dset = f.create_dataset("table", table.shape, data=table)

# muKL approach
else:

    for rpair in reaction_pairs:

        rref = rpair[0]

        mean[rref] = {}
        eigval[rref] = {}
        eigvec[rref] = {}
        scores[rref] = {}

        # breakpoint()
        for rate in lumped_rates:

            # Concatenate forward and backward rates
            fwd_bkw = np.append(full_rate[rpair[0]][rate], full_rate[rpair[1]][rate], axis=0)
            work = fwd_bkw
            if log_model:
                work = np.log(fwd_bkw)
            mean[rref][rate], eigval[rref][rate], eigvec[rref][rate], scores[rref][rate] = estimateCovarianceEig(work)


        if rref == "Ionization":
            fwd_bkw = np.append(full_rate[rpair[0]]["Ground"], full_rate[rpair[1]]["Ground"], axis=0)

            work = fwd_bkw
            if log_model:
                fwd_bkw[0] = max(fwd_bkw[0,0], 1e-300)
                work = np.log(fwd_bkw)
                # breakpoint()
            mean[rref]["Ground"], eigval[rref]["Ground"], eigvec[rref]["Ground"], scores[rref]["Ground"] = estimateCovarianceEig(work)


    # stepwise excitation
    if sample_step_exc:
        rref = "StepExcitation"
        mean[rref] = {}
        eigval[rref] = {}
        eigvec[rref] = {}
        scores[rref] = {}
        # for rate_i in lumped_rates:
        for i in range(len(lumped_rates) - 1):
            # for rate_j in lumped_rates:
            for j in range(i, len(lumped_rates)):
                rate_i = lumped_rates[i]
                rate_j = lumped_rates[j]
                if rate_i == rate_j:
                    continue

                rate_fwd = f'{rate_i}_{rate_j}'
                rate_bkw = f'{rate_j}_{rate_i}'

                # Concatenate forward and backward rates
                fwd_bkw = np.append(full_rate[rref][rate_fwd], full_rate[rref][rate_bkw], axis=0)

                work = fwd_bkw
                if log_model:
                    work = np.log(fwd_bkw)
                mean[rref][rate_fwd], eigval[rref][rate_fwd], eigvec[rref][rate_fwd], scores[rref][rate_fwd] = estimateCovarianceEig(work)# breakpoint()


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


# NOTE: Update this, and the calculation
# compute variance percentage too
frac = {}
# r1 = {}
# r2 = {}
if indep_approach:

    for rtype in mean.keys():
        frac[rtype] = {}
        # r1[rtype] = {}
        # r2[rtype] = {}
    
        lumped_rates_check = lumped_rates
        if rtype == "Ionization" or rtype == "Recombination":  
            lumped_rates_check = lumped_rates_g
    
        for rate in lumped_rates_check:
            print("{0} Indices: ".format(rate))
            frac[rtype][rate] = computeVarianceFractions(eigval[rtype][rate])
            print(frac[rtype][rate])

# muKL approach
else:

    for rpair in reaction_pairs:

        rref = rpair[0]
        print(rref)
        frac[rref] = {}
        lumped_rates_check = lumped_rates
        if rref == "Ionization":  
            lumped_rates_check = lumped_rates_g

        for rate in lumped_rates_check:
            print("{0} Indices: ".format(rate))
            frac[rref][rate] = []
            frac[rref][rate].append(computeVarianceFractions(eigval[rref][rate])) # both fwd and bkw

            # fwd spectrum
            fwd_val = eigval[rref][rate] + 0.
            for k in range(fwd_val.shape[0]):
                fwd_val[k] *= np.linalg.norm(eigvec[rref][rate][k][:N_T])**2
            frac[rref][rate].append(computeVarianceFractions(fwd_val))
            
            # bkw spectrum
            bkw_val = eigval[rref][rate] + 0.
            for k in range(bkw_val.shape[0]):
                bkw_val[k] *= np.linalg.norm(eigvec[rref][rate][k][N_T:])**2
            frac[rref][rate].append(computeVarianceFractions(bkw_val))

            print(frac[rref][rate])

    # stepwise excitation
    if sample_step_exc:
        rref = "StepExcitation"
        print(rref)
        frac[rref] = {}
        for i in range(len(lumped_rates) - 1):
            # for rate_j in lumped_rates:
            for j in range(i, len(lumped_rates)):
                rate_i = lumped_rates[i]
                rate_j = lumped_rates[j]
                if rate_i == rate_j:
                    continue

                rate_fwd = f'{rate_i}_{rate_j}'
                rate_bkw = f'{rate_j}_{rate_i}'
                
                print("{0} Indices: ".format(rate_fwd))
                frac[rref][rate_fwd] = []
                frac[rref][rate_fwd].append(computeVarianceFractions(eigval[rref][rate_fwd])) # both fwd and bkw

                # fwd spectrum
                fwd_val = eigval[rref][rate_fwd] + 0.
                for k in range(fwd_val.shape[0]):
                    fwd_val[k] *= np.linalg.norm(eigvec[rref][rate_fwd][k][:N_T])**2
                frac[rref][rate_fwd].append(computeVarianceFractions(fwd_val))
                
                # bkw spectrum
                bkw_val = eigval[rref][rate_fwd] + 0.
                for k in range(bkw_val.shape[0]):
                    bkw_val[k] *= np.linalg.norm(eigvec[rref][rate_fwd][k][N_T:])**2
                frac[rref][rate_fwd].append(computeVarianceFractions(bkw_val))

                print(frac[rref][rate_fwd])
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
with open(res_dir + "/is_log.pickle", 'wb') as f: #KL on log of rates
    pickle.dump(log_model ,f)
with open(res_dir + "/is_indep.pickle", 'wb') as f: #separate 
    pickle.dump(indep_approach ,f)
# with open(res_dir + "/r1.pickle", 'wb') as f:
#     pickle.dump(r1 ,f)
# with open(res_dir + "/r2.pickle", 'wb') as f:
#     pickle.dump(r2 ,f)


# breakpoint()





