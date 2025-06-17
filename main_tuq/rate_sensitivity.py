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
Script for sensitivity analysis of rate expansion coefficients to torch1d outputs

Run after: 

resample_rates.py

**Run the torch1d samples**

sf_torch1d_uq_post.py
"""




home = os.getenv('HOME')

### Input and Output Directories
in_dir = home + "/bedonian1/torch1d_resample_sens_r8/"
out_dir = home + "/bedonian1/torch1d_post_sens_r8/"

### Reaction Types to analyze
reaction_types_fwd = ['Excitation', 'Ionization', 'StepExcitation']

### Plot Options
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 16,
})
reaction_types_fwd_to_label = {
    'Excitation': 'Excitation', 
    'Ionization': 'Ionization', 
    'StepExcitation': 'Step. Exc.'
}
prate_to_label = {
    'Ground': r'Ground',
    'meta': r'Ar$^m$',
    'res': r'Ar$^r$',
    'fourp': r'Ar$^{4p}$',
    'higher': r'Ar$^h$',
    'meta_res': r'Ar$^m$ to Ar$^r$',
    'meta_fourp': r'Ar$^m$ to Ar$^{4p}$',
    'meta_higher': r'Ar$^m$ to Ar$^h$',
    'res_fourp': r'Ar$^r$ to Ar$^{4p}$',
    'res_higher': r'Ar$^r$ to Ar$^h$',
    'fourp_higher': r'Ar$^{4p}$ to Ar$^h$'
    
}
qoi_to_label = {
    'exit_v': "Exit Velocity",
    'exit_X': "Exit Fraction",
    'exit_T': "Exit Temperature",
    'exit_X': "Exit Fraction",
    'exit_p': "Exit Pressure",
    'exit_d': "Exit Density",
    'heat_dep': "Heat Deposition"
}

### Group Prefix, shouldn't change
groups = ['A', 'B', 'AB']


##########################################################################################################
# Script Starts Here
##########################################################################################################

perturb_samples_A = {}
perturb_samples_B = {}

# load in inputs to get variable info
for rtype in reaction_types_fwd:

    a_sample_file = in_dir + rtype + "_perturb_samples_A.pickle"
    b_sample_file = in_dir + rtype + "_perturb_samples_B.pickle"

    with open(a_sample_file, 'rb') as f:
        perturb_samples_A[rtype] = pickle.load(f)
    with open(b_sample_file, 'rb') as f:
        perturb_samples_B[rtype] = pickle.load(f)

# number of principal components to examine in sensitivity analysis
N = perturb_samples_A['Excitation'].data['meta'].shape[1]
N_pc = {}
Nvars = 0
for rtype in reaction_types_fwd:
    N_pc[rtype] = {}
    for rate in perturb_samples_A[rtype].data.keys():
        N_pc[rtype][rate] = perturb_samples_A[rtype].data[rate].shape[0]
        Nvars += perturb_samples_A[rtype].data[rate].shape[0]

# breakpoint()

# load outputs
qoi_val = {}
fqoin = out_dir + '/qoi_list.pickle' 
with open(fqoin, 'rb') as f:
    qoi_list = pickle.load(f)

for group in groups:

    fqoiv = out_dir + f'/qoi_samples_{group}.pickle' 

    with open(fqoiv, 'rb') as f:
        qoi_val[group] = pickle.load(f)
        
    # # # remove zero entries
    # # mask[res] = np.nonzero(qoi_val[qoi_list[0]][:,0])

    # for qoi in qoi_list:
    #     qoi_val[qoi] = qoi_val[qoi][mask[res],:][0]
    # # breakpoint()

    # qoi_vals[res] = qoi_val

#Estimate Sobol Indices
# breakpoint()
# compute variance percentage too
r1 = {}
r2 = {}

for qoi in qoi_list:
    # r1[qoi] = {}
    # r2[qoi] = {}

    r1[qoi], r2[qoi] = computeSobolIndices2(qoi_val['A'][qoi].T, qoi_val['B'][qoi].T, qoi_val['AB'][qoi].T, Nvars)
    
    print(qoi)
    print(r1[qoi])
    print(r2[qoi])
    print()
    r1[qoi] = np.atleast_2d(r1[qoi]).T
    r2[qoi] = np.atleast_2d(r2[qoi]).T


# breakpoint()

if not os.path.isdir(out_dir + '/plots'):
    os.makedirs(out_dir + '/plots')

# bar plots of sensitivities
for qoi in qoi_list:
    for j in range(r1[qoi].shape[1]):
        fig, ax = plt.subplots(layout='constrained')
        fig.set_figheight(6.5)

        fig.set_figwidth(7)

        # non zero sensitivities
        # sens_ind = np.nonzero(r1[qoi][0,:])[0]

        x = np.arange(len(r1[qoi]))
        width = 0.30
        # if prate == 'higher':
        #     width = 0.51
        mult = 0

        xl = []

        # assuming consistent order
        c = 0
        ticks_abr = []
        for ptype in perturb_samples_A.keys():
            for prate in perturb_samples_A[ptype].data.keys():
                # mult = 0
                for i in range(N_pc[ptype][prate]):
                    # offset = width*mult
                    # ax.bar_label(rect, padding=N_pc)
                    # xl.append(np.mean(x) + offset)

                    ticks_abr.append(f"{reaction_types_fwd_to_label[ptype]} {prate_to_label[prate]} {i}")
                    # mult += 1
                    c += 1
        # breakpoint()
        rect = ax.barh(x, r1[qoi][:,j], width)#,  label = config_list[sens_ind%config_list.shape[0]])
        plt.xlabel(rf"Sensitivity")
        # plt.grid()
        # plt.legend()

        # for i in sens_ind:
            # ticks.append(config_list[i%config_list.shape[0]])
        # ax.set_yticks(x, ticks)

        ax.set_yticks(x, ticks_abr)
        # plt.yticks)
        # plt.title(f"{qoi} {j} Sensitivity Indices")
        plt.title(f"{qoi_to_label[qoi]} Sensitivity Indices")
        plt.savefig(out_dir + f"plots/argon-rate-{qoi}-{j}-sens-KL.png", dpi=600, bbox_inches='tight')
        plt.clf()
