import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

res_dir = "results/test_stepwise_2/"


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
    scores = pickle.load(f)
with open(res_dir + "/r1.pickle", 'rb') as f:
    r1 = pickle.load(f)
with open(res_dir + "/r2.pickle", 'rb') as f:
    r2 = pickle.load(f)

reaction_types = list(mean.keys())

known_configurations = ['4s', '5s', '6s',
                        '4p', '5p', '6p',
                        '3d', '4d', '5d', '6d']

df = pd.read_csv('~/torch-chemistry/argon/input-data/ArI-levels-nist.csv')
configuration = df['Configuration']
term = df['Term']
J = df['J']

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

for ptype in reaction_types:
    for prate in lumped_rates:

        fig, ax = plt.subplots(layout='constrained')
        fig.set_figheight(10)
        fig.set_figwidth(9)

        # non zero sensitivities
        sens_ind = np.nonzero(r1[ptype][prate][0,:])[0]

        x = np.arange(len(sens_ind))
        width = 0.20
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
        
        for i in sens_ind:
            ticks.append(config_list[i%90])
        ax.set_yticks(x, ticks)
        # plt.yticks)
        plt.title(f"Argon {ptype} {prate} Sensitivity")
        plt.savefig(res_dir + f"plots/argon-{ptype}-{prate}-config-sens-KL.pdf", bbox_inches='tight')
        plt.clf()
