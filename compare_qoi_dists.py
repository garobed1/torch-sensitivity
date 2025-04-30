import numpy as np
from scipy.special import kl_div
from scipy.stats import gaussian_kde
import os, sys
import matplotlib.pyplot as plt
import pickle

home = os.getenv('HOME')
res_names = ['xsection samples', 'KL model resamples']
res_dirs = [home + "/bedonian1/torch1d_post_r7/", home + "/bedonian1/torch1d_re_post_r7/"]


kde = True

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 22,
})

mask = {}
qoi_vals = {}
for r, res in enumerate(res_names):

    res_dir = res_dirs[r]
    fqoiv = res_dir + '/qoi_samples.pickle' 
    fqoin = res_dir + '/qoi_list.pickle' 

    with open(fqoiv, 'rb') as f:
        qoi_val = pickle.load(f)
        
    with open(fqoin, 'rb') as f:
        qoi_list = pickle.load(f)


    # remove zero entries
    mask[res] = np.nonzero(qoi_val[qoi_list[0]][:,0])

    # if r == 1:
    #     mask[res] = np.nonzero(qoi_val[qoi_list[0]][:4000,0])

    for qoi in qoi_list:
        qoi_val[qoi] = qoi_val[qoi][mask[res],:][0]
    # breakpoint()

    qoi_vals[res] = qoi_val
# breakpoint()
    # for qoi in fqoin:
        
# start with histogram

# OVERRIDE
qoi_list = ['exit_p', 'exit_T', 'exit_X']

nFig = 0
qoi_num = {}
for qoi in qoi_list:
    # qoi_num[qoi] = qoi_val[qoi].shape[1]
    qoi_num[qoi] = 1
    nFig += qoi_num[qoi]

fig, axs = plt.subplots(nFig, figsize=(12,5. * nFig))


qoi_labels = {
    "exit_p": ['Exit Pressure', r'$p$ ($Pa$)', r'$P(p)$'],
    "exit_d": ['Exit Density', r'$\rho$ ($kg/m^3$)', r'$P(\rho)$'],
    "exit_v": ['Exit Axial Velocity', r'$u_z$ ($m/s$)', r'$P(u_z)$'],
    "exit_T": ['Exit Temperature', r'$T$ ($K$)', r'$P(T)$'],
    "exit_X": ['Exit Mole Fraction (log)', r'$X$', r'$P(X)$'],
    "heat_dep": ['Heat Deposition', r'$q_e$ ($kW/m$)', r'$P(q_e)$']
}

qoi_legends = {
    "exit_p": [''],
    "exit_d": ['(Ar)', '(e)'],
    "exit_v": [''],
    "exit_T": ['(Ar)', '(e)'],
    "exit_X": ['(Ar$^+$)', '(Ar$^m$)', '(Ar$^r$)', '(Ar$^p$)', '(Ar$^h$)' ],
    "heat_dep": ['']
}

dist_label = {
    0: "Cross-section propagation",
    1: "Rate model propagation",
}

cq = 0
for qoi in qoi_list:

    for i in range(qoi_num[qoi]):
    # for i in range(1):
        bins = 20
        qkde = {}
        kde_space = np.linspace(min(qoi_vals[res][qoi][:,i]), max(qoi_vals[res][qoi][:,i]), 1000)
        print(qoi)
        rq = 0
        for res in res_names:
            if kde:
                qkde[res] = gaussian_kde(qoi_vals[res][qoi][:,i])
                yval = qkde[res].evaluate(kde_space)
                axs[cq].plot(kde_space, qkde[res].evaluate(kde_space), label=dist_label[rq])
                print(np.trapz(yval))
            else:
                counts, bins, _ = axs[cq].hist(qoi_vals[res][qoi][:,i], 
                                    bins=bins, density=True, label=res, alpha = 1.0/len(res_names))
            rq += 1
        # breakpoint()
    #     axs[0].plot(solver.grid.xg, div[x], '-b')

        kl_val = ''
        if kde:
            kl_val = np.sum(kl_div(qkde[res_names[0]].evaluate(kde_space), qkde[res_names[1]].evaluate(kde_space)))/np.sum(qkde[res_names[0]].evaluate(kde_space))

        axs[cq].grid()
        axs[cq].set_title(qoi_labels[qoi][0] + ' ' + qoi_legends[qoi][i] + ', Rel. Div. = {:.4f}'.format(kl_val))
        axs[cq].set_xlabel(qoi_labels[qoi][1])
        axs[cq].set_ylabel(qoi_labels[qoi][2])

        if cq == 0:
            axs[cq].legend(fontsize=18)

        cq += 1
    
# breakpoint()


fig.tight_layout()

if kde:
    # fig.savefig(f'kpdes_full.png')
    fig.savefig(f'kpdes_pres.png')
else:
    fig.savefig(f'hists_full.png')




# KL divergence 