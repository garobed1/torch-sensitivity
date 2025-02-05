import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os

import bsr.excitation_cs as bsr_exc
import bultel.excitation_cs as bul
import deutsch.ionization_cs as dmi
import bsr.ionization_cs as bsr_ion


home = os.getenv('HOME')
sample_dir = home + "/torch-chemistry/argon/results/test_stepwise_2/"
res_dir = "results/test_stepwise_2/"
plt.rcParams["font.size"] = 16

# number of principal components to examine in sensitivity analysis
N_T = 512
N_pc = 1


# NOTE: excluding step exc for now


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



with open(res_dir + "/mean.pickle", 'rb') as f:
    mean = pickle.load(f)
with open(res_dir + "/eigval.pickle", 'rb') as f:
    eigval = pickle.load(f)
with open(res_dir + "/eigvec.pickle", 'rb') as f:
    eigvec = pickle.load(f)
with open(res_dir + "/scores.pickle", 'rb') as f:
    scores = pickle.load(f)
with open(res_dir + "/frac.pickle", 'rb') as f:
    frac = pickle.load(f)
with open(res_dir + "/r1.pickle", 'rb') as f:
    r1 = pickle.load(f)
with open(res_dir + "/r2.pickle", 'rb') as f:
    r2 = pickle.load(f)

# nominal cross sections
with open(sample_dir + '/argon_excitation_sigma.pickle', 'rb') as f:
    sigma_exc = pickle.load(f)
with open(sample_dir + '/argon_ionization_sigma.pickle', 'rb') as f:
    sigma_ion = pickle.load(f)
            
sig_exc_sum = []
sig_ion_sum = []


reaction_types = list(mean.keys())

known_configurations = ['4s', '5s', '6s',
                        '4p', '5p', '6p',
                        '3d', '4d', '5d', '6d']

df = pd.read_csv('~/torch-chemistry/argon/input-data/ArI-levels-nist.csv')
configuration = df['Configuration']
term = df['Term']
J = df['J']
energy_level = df['Level (eV)'].to_numpy()

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

config_list = np.array(list(config_perturb_dist.keys()))
# breakpoint()

for i in config_list:
    sig_exc_sum.append(sum(sigma_exc[i]))
    sig_ion_sum.append(sum(sigma_ion[i]))

sig_sum = {"Excitation":np.array(sig_exc_sum),
           "Ionization":np.array(sig_ion_sum),
           }
# breakpoint()


# Grid in energy (where cross-sections will be evaluated and then used to approximate integrals)
egrid = np.logspace(1, np.log10(300), 5001) # eV

# NOTE: Ordering of species is already in 
for ptype in reaction_types:


    # plot nominal cross sections for this reaction type

    if ptype == 'Excitation':

        sigma_bsr = bsr_exc.evaluate_excitation_cross_sections(df, egrid, f"{home}/torch-chemistry/argon/input-data/bsr_argon_ground_excitation.txt")
        sigma_bul = bul.evaluate_excitation_cross_sections(df, egrid)

        # Merge cross sections---i.e., if we have BSR cross section, use it.  Otherwise, use Bultel.
        sigma = {}
        for key in sigma_bul.keys():
            if (key in sigma_bsr.keys()):
                sigma[key] = sigma_bsr[key]
            else:
                sigma[key] = sigma_bul[key]

        inds = np.arange(0, config_list.shape[0])

        # for i in inds:
        for i in range(0, 90):
            plt.loglog(egrid, sigma[config_list[i]], color=plt.cm.RdYlBu(i/config_list.shape[0]))
        plt.xlabel(rf'$\epsilon$')
        plt.ylabel(rf'$\sigma(\epsilon)$')
        # plt.colorbar()
        plt.savefig(res_dir + f"plots/excitation_sigma_nominal.pdf", bbox_inches='tight')


    if ptype == 'Ionization':

        sigma = dmi.evaluate_ionization_cross_sections(df, egrid)
        # Merge cross sections---i.e., if we have BSR cross section, use it.  Otherwise, use Bultel.

        inds = np.arange(0, config_list.shape[0])

        # for i in inds:
        for i in range(0, 90):
            plt.loglog(egrid, sigma[config_list[i]], color=plt.cm.RdYlBu(i/config_list.shape[0]))
        plt.xlabel(rf'$\epsilon$')
        plt.ylabel(rf'$\sigma(\epsilon)$')
        # plt.colorbar()
        plt.savefig(res_dir + f"plots/ionization_sigma_nominal.pdf", bbox_inches='tight')

    # breakpoint()

    for prate in lumped_rates:

        fig, ax = plt.subplots(layout='constrained')
        fig.set_figheight(6.5)
        if prate == 'higher':
            fig.set_figheight(9.5)
        fig.set_figwidth(7)

        # non zero sensitivities
        sens_ind = np.nonzero(r1[ptype][prate][0,:])[0]

        x = np.arange(len(sens_ind))
        width = 0.30
        # if prate == 'higher':
        #     width = 0.51
        mult = 0

        xl = []
        for i in range(N_pc):
            offset = width*mult
            rect = ax.barh(x + offset,  abs(r1[ptype][prate][i,sens_ind]), width,  label = config_list[sens_ind%90])
            # ax.bar_label(rect, padding=N_pc)
            # xl.append(np.mean(x) + offset)

            mult += 1

        plt.xlabel(rf"Sensitivity")
        # plt.grid()
        # plt.legend()
        ticks = []
        ticks_abr = []

        for i in sens_ind:
            ticks.append(config_list[i%90])
            ticks_abr.append(config_list[i%90][19:])
        # ax.set_yticks(x, ticks)
        if prate == 'higher':
            ax.set_yticks(x, ticks_abr, fontsize='9')
        else:
            ax.set_yticks(x, ticks_abr)
        # plt.yticks)
        # plt.title(f"Argon {ptype} {prate} Sensitivity")
        plt.savefig(res_dir + f"plots/argon-{ptype}-{prate}-config-sens-KL.pdf", bbox_inches='tight')
        plt.clf()


        # We also want to plot the magnitude of the cross sections to normalize the sensitivities
        # e.g. the species is sensitive but its magnitude is small
        # other integral terms are same for all rate coefficients, so we can look at just sigma
        # pull up nominal data
        if ptype in sig_sum:
            fig, ax = plt.subplots(layout='constrained')
            fig.set_figheight(10)
            fig.set_figwidth(9)

            # non zero sensitivities

            x = np.arange(len(sens_ind))
            width = 0.20
            mult = 0

            xl = []
            rect = ax.barh(x,  sig_sum[ptype][sens_ind%90], width,  label = config_list[sens_ind%90])
            # ax.bar_label(rect, padding=N_pc)
            # xl.append(np.mean(x) + offset)


            plt.xlabel(rf"Magnitude")
            # plt.grid()
            # plt.legend()
            ticks = []
            ticks_abr = []

            for i in sens_ind:
                ticks.append(config_list[i%90])
                ticks_abr.append(config_list[i%90][19:])

            # ax.set_yticks(x, ticks)
            ax.set_yticks(x, ticks_abr)
            # plt.yticks)
            plt.title(f"Argon {ptype} {prate} Cross-Section Magnitude")
            plt.savefig(res_dir + f"plots/argon-{ptype}-{prate}-sigma-mag-KL.pdf", bbox_inches='tight')
            plt.clf()