import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import scipy.constants as spc
import pickle
from mpi4py import MPI

from rate_utils import *
from sample_utils import *
from sobol_tools import *
from pca_tools import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Equivalent to sample_cross_sections.py in torch-chemistry, except we get forward rates
by resampling KL models, then compute backward rates
"""

#NOTE: Trying 4 species now

home = os.getenv('HOME')
# kl_model_dir = 'results/rate_resample_r7/'
kl_model_dir = home + "/bedonian1/rate_resample_model_r8/"
# kl_model_dir = home + "/bedonian1/rate_resample_model_4s_r8/"
# nom_dir = home + "/torch-sensitivity/trevilo-cases/torch_7sp_chem/nominal/rate-coefficients/"
# nom_dir = home + "/mean_4s_r6/"
nom_dir = home + "/mean_r6/"
# res_dir = home + "/bedonian1/torch1d_resample_sens_r7/"
# res_dir = home + "/bedonian1/rate_mf_r1_pilot/"
# res_dir = home + "/bedonian1/rate_mf_r1_pilot_4s/"
res_dir = home + "/bedonian1/rate_mf_r1_G1/"
# res_dir = home + "/bedonian1/rate_mf_r1_G2/"
# res_dir = home + "/bedonian1/rate_mf_r1_G3/"
# res_dir = home + "/bedonian1/rate_mf_r1_G4/"

four_species = False
# four_species = True

#### SAMPLING INPUTS ####
# Generate some artificial distributions for every considered cross section
Nsamples = 50000
# Nsamples = 100
# Nsamples = 12
# Nsamples = 1024 # for Sobol sensitivity
# Nsamples = 256 # tentative covariance estimation sample 
# Nsamples = 4

N_T = 512
pc_threshold = 1.0 - 1.e-2 #coarse
# pc_threshold = 1.0 - 1.e-4 
pc_proc = 0 # if muKL, 0 means both processes, 1 onward places threshold on that process' spectrum


# sample_exc = True
# sample_ion = True
# sample_step_exc = False

make_plots = False
# compute_sobol = False
compute_sobol = False

if four_species:
    reaction_types_fwd = ['Excitation', 'Ionization']
    lumped_rates = ["Lumped"]
    lumped_rates_g = ["Lumped", "Ground"]
    reaction_types_bwd = ['Excitation', 'Deexcitation', 'Ionization', 'Recombination']
else:
    reaction_types_fwd = ['Excitation', 'Ionization', 'StepExcitation']
    lumped_rates = ["meta", "res", "fourp", "higher"]
    lumped_rates_g = ["meta", "res", "fourp", "higher", "Ground"]
    lumped_rates_fwd_step = ["meta_res", "meta_fourp", "meta_higher", 
                            "res_fourp", "res_higher", "fourp_higher"]
    lumped_rates_bwd_step = ["res_meta", "fourp_meta", "higher_meta", 
                            "fourp_res", "higher_res", "higher_fourp"]
    reaction_types_bwd = ['Excitation', 'Deexcitation', 'Ionization', 'Recombination']


sfwd_to_bkw = {
    "meta_res": "res_meta", 
    "meta_fourp": "fourp_meta",
    "meta_higher": "higher_meta",
    "res_fourp": "fourp_res",
    "res_higher": "higher_res",
    "fourp_higher": "higher_fourp"
}

# determine number of principal components/independent variables for each sample based on eigval threshold
# pc_threshold = 1.0 - 1.e-4

#### END INPUTS ####

if rank == 0:
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

with open(kl_model_dir + "/mean.pickle", 'rb') as f:
    mean = pickle.load(f)
with open(kl_model_dir + "/eigval.pickle", 'rb') as f:
    eigval = pickle.load(f)    
with open(kl_model_dir + "/eigvec.pickle", 'rb') as f:
    eigvec = pickle.load(f)    
with open(kl_model_dir + "/scores.pickle", 'rb') as f:
    scores = pickle.load(f)    
with open(kl_model_dir + "/frac.pickle", 'rb') as f:
    frac = pickle.load(f)    
with open(kl_model_dir + "/is_log.pickle", 'rb') as f:
    is_log = pickle.load(f)    
with open(kl_model_dir + "/is_indep.pickle", 'rb') as f:
    is_indep = pickle.load(f)    
 
# determine number of vars per, should be given solely by pc_threshold


N_pc = {}
if not is_indep:
    for rtype in reaction_types_fwd:
        N_pc[rtype] = {}
        for rate in frac[rtype].keys():
            c = 0
            N_pc[rtype][rate] = []
            for proc in frac[rtype][rate]:
                while proc[c] < pc_threshold:
                    c += 1
                # at least one var
                N_pc[rtype][rate].append(max(c + 0, 1))

#NOTE: Hard code ground
N_pc['Ionization']['Ground'] = [2, 2, 2]

print(N_pc)
# breakpoint()

qe = spc.e    # 1.60217663e-19 [C]
kB = spc.k    # 1.380649e-23 [J/K]
NA = spc.N_A  # 6.022e23 [#/mol]
eV = qe/kB    # 11604.518 [K / eV]


# temp grid
T_Maxw = np.linspace(0.02, 2, N_T) * eV






config_perturb_dist = {}
sizes = {}
Nvars = {}
# for i in range(0,len(df)):
#     base_config = configuration[i][19:]
#     if base_config in known_configurations:

#NOTE: The bayesian samples, the first 7 or so may be way off, need to recompute
###############################################################################################
# Characterize uncertainties
for rtype in reaction_types_fwd:


    if rtype == "Excitation":
        lumped_rates_check = lumped_rates
    if rtype == "Ionization":
        lumped_rates_check = lumped_rates_g
    if rtype == "StepExcitation":
        lumped_rates_check = lumped_rates_fwd_step
    
    Nvars[rtype] = 0
    sizes[rtype] = {}
    config_perturb_dist[rtype] = {}

    for rate in lumped_rates_check:
        # Nconfig
        

        #NOTE: N_pc number of normally distributed indep vars
        sizes[rtype][rate] = N_pc[rtype][rate][pc_proc]
        Nvars[rtype] += sizes[rtype][rate]
        config_perturb_dist[rtype][rate] = {"dist":"normal", "model":False, "loc":0., 
                                            "scale": 1.}





NvarsTotal = 0
for key, item in Nvars.items():
    NvarsTotal += item



################################################################################################
# Sample Generation
# Ground Excitation

perturb_samples_base = {}
perturb_samples_A = {}
perturb_samples_B = {}
perturb_samples_AB = {}

for rtype in config_perturb_dist.keys():

    perturb_samples_base_r0 = None
    if rank == 0:
        perturb_samples_base_r0 = SampleData(categories=list(config_perturb_dist[rtype].keys()),
                                        sizes=sizes[rtype], scales=config_perturb_dist[rtype])
        perturb_samples_base_r0.createData(N=Nsamples)
    
    perturb_samples_base[rtype] = comm.bcast(perturb_samples_base_r0, root=0)

    a_sample_file = res_dir + rtype + "_perturb_samples_A.pickle"
    b_sample_file = res_dir + rtype + "_perturb_samples_B.pickle"

    perturb_samples_A[rtype] = None
    perturb_samples_B[rtype] = None
    perturb_samples_AB[rtype] = None

    if rank == 0:
        if os.path.exists(a_sample_file):
            if compute_sobol:
                with open(a_sample_file, 'rb') as f:
                    perturb_samples_A[rtype] = pickle.load(f)
                with open(b_sample_file, 'rb') as f:
                    perturb_samples_B[rtype] = pickle.load(f)
                perturb_samples_A[rtype], perturb_samples_B[rtype], perturb_samples_AB[rtype] = perturb_samples_A[rtype].genSobolData(perturb_samples_B[rtype])


            else:
                with open(a_sample_file, 'rb') as f:
                    perturb_samples_A[rtype] = pickle.load(f)   
            # if "StepExcitation" in rtype:
            #     breakpoint()
            # perturb_samples_A[rtype] = comm.bcast(perturb_samples_A[rtype], root=0)

        else:
            if compute_sobol:
                perturb_samples_A_r0, perturb_samples_B_r0, perturb_samples_AB_r0 = perturb_samples_base[rtype].genSobolData()

                perturb_samples_A[rtype], perturb_samples_B[rtype], perturb_samples_AB[rtype] = perturb_samples_base[rtype].genSobolData(perturb_samples_B[rtype])

                # if rank == 0:
                with open(a_sample_file, 'wb') as f:
                    pickle.dump(perturb_samples_A[rtype], f)
                with open(b_sample_file, 'wb') as f:
                    pickle.dump(perturb_samples_B[rtype], f)

            else:
                perturb_samples_A[rtype] = perturb_samples_base[rtype]
                # if rank == 0:
                with open(a_sample_file, 'wb') as f:
                    pickle.dump(perturb_samples_A[rtype], f)
                # comm.Barrier()
    perturb_samples_A[rtype] = comm.bcast(perturb_samples_A[rtype], root=0)
    perturb_samples_B[rtype] = comm.bcast(perturb_samples_B[rtype], root=0)
    # perturb_samples_AB[rtype] = comm.bcast(perturb_samples_AB[rtype], root=0)

#################################################################################################
# Propagation

# divide sample cases
groups = ['A']
if compute_sobol:
    groups = ['A', 'B', 'AB']

perturb_samples = {}

for rtype in config_perturb_dist.keys():

    perturb_samples[rtype] = {'A': perturb_samples_A[rtype]}

    if compute_sobol:
        perturb_samples[rtype]['B'] = perturb_samples_B[rtype]
        perturb_samples[rtype]['AB'] = perturb_samples_AB[rtype]



for group in groups:

    if group == 'AB':
        cases = divide_cases(Nsamples*NvarsTotal, size)
    else:
        cases = divide_cases(Nsamples, size)

    ### TEST
    # cases[rank] = [28*4 + 15*4 + 13*4 + 2]

    for isamp in cases[rank]:
    # for isamp in [2]:

        # A place for plots
        results_dir = "{0}sig_{1}_{2:06d}/plots".format(res_dir, group, isamp)


        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        # Place to write data
        rate_dir = "{0}sig_{1}_{2:06d}/rates".format(res_dir, group, isamp)
        if not os.path.isdir(rate_dir):
            os.makedirs(rate_dir)
        
        # Avoid recomputing if we can
        # if os.path.exists(results_dir):
        if checkRateSample(rate_dir, True, True, True):
            continue
        
        group_exc = group
        group_ion = group
        group_step_exc = group

        # map out appropriate samples from each reaction group
        ind_samp_dict, group_dict = mapABReaction2(isamp, Nsamples, group, Nvars, NvarsTotal)

        # assemble perturbations
        # exc_sample = perturb_samples["Excitation"]['A'](isamp)
        exc_sample = perturb_samples["Excitation"][group_dict['Excitation']](ind_samp_dict['Excitation'])
        

        # Excitation
        rtype = 'Excitation'
        table = np.zeros((len(T_Maxw), 2))
        for key in exc_sample.keys():
            # Excitation rates

            if group_dict[rtype] == 'AB':
                exc_sample[key] = exc_sample[key].T[0]

            N_pc = len(exc_sample[key])

            krate = mean[rtype][key] + np.dot(exc_sample[key]*np.sqrt(eigval[rtype][key][:N_pc]),
                                                        eigvec[rtype][key][:,:N_pc].T)

            kexcite_fwd = krate[:N_T]
            kexcite_bkw = krate[N_T:]

            if is_log:
                kexcite_fwd = np.exp(kexcite_fwd)
                kexcite_bkw = np.exp(kexcite_bkw)
            

            # breakpoint()
            fname = os.path.join(rate_dir, 'Excitation_' + key + '.h5')
            table[:,0] = T_Maxw
            table[:,1] = kexcite_fwd
            with h5.File(fname, 'w') as f:
                f.attrs['process'] = 'Ar(Ground) + e -> Ar('+key+') + e'
                f.attrs['temperature units'] = 'K'
                f.attrs['rate units'] = 'm^3 / mol / s'
                dset = f.create_dataset("table", table.shape, data=table)

            # Deexcitation rates
            fname = os.path.join(rate_dir, 'Deexcitation_' + key + '.h5')
            table[:,0] = T_Maxw
            table[:,1] = kexcite_bkw
            with h5.File(fname, 'w') as f:
                f.attrs['process'] = 'Ar('+key+') + e -> Ar(Ground) + e'
                f.attrs['temperature units'] = 'K'
                f.attrs['rate units'] = 'm^3 / mol / s'
                dset = f.create_dataset("table", table.shape, data=table)
    
        # Ionization

        
        
        # assemble perturbations
        # ion_sample = perturb_samples["Ionization"]['A'](isamp)
        ion_sample = perturb_samples["Ionization"][group_dict["Ionization"]](ind_samp_dict["Ionization"])
        
        # Forward ionization rates
        rtype = 'Ionization'
        table = np.zeros((len(T_Maxw), 2))
        for key in ion_sample.keys():

            if group_dict[rtype] == 'AB':
                ion_sample[key] = ion_sample[key].T[0]

            N_pc = len(ion_sample[key])

            krate = mean[rtype][key] + np.dot(ion_sample[key]*np.sqrt(eigval[rtype][key][:N_pc]),
                                                        eigvec[rtype][key][:,:N_pc].T)

            kion_fwd = krate[:N_T]
            kion_bkw = krate[N_T:]

            if is_log:
                kion_fwd = np.exp(kion_fwd)
                kion_bkw = np.exp(kion_bkw)


            fname = os.path.join(rate_dir, 'Ionization_' + key + '.h5')
            table[:,0] = T_Maxw
            table[:,1] = kion_fwd
            with h5.File(fname, 'w') as f:
                f.attrs['process'] = 'Ar('+key+') + e -> Ar+ + e + e'
                f.attrs['temperature units'] = 'K'
                f.attrs['rate units'] = 'm^3 / mol / s'
                dset = f.create_dataset("table", table.shape, data=table)


            fname = os.path.join(rate_dir, 'Recombination_' + key + '.h5')
            table[:,0] = T_Maxw
            table[:,1] = kion_bkw
            with h5.File(fname, 'w') as f:
                f.attrs['process'] = 'Ar+ + e + e -> Ar('+key+') + e'
                f.attrs['temperature units'] = 'K'
                f.attrs['rate units'] = 'm^6 / mol^2 / s'
                dset = f.create_dataset("table", table.shape, data=table)


        # Stepwise Excitation


        # assemble_perturbations
        # step_exc_sample = perturb_samples["StepExcitation"]['A'](isamp)

        if "StepExcitation" in reaction_types_fwd:
            step_exc_sample = perturb_samples["StepExcitation"][group_dict["StepExcitation"]](ind_samp_dict["StepExcitation"])

            rtype = 'StepExcitation'
            table = np.zeros((len(T_Maxw), 2))
            
            for key in step_exc_sample.keys():

                if group_dict[rtype] == 'AB':
                    step_exc_sample[key] = step_exc_sample[key].T[0]

                N_pc = len(step_exc_sample[key])

                krate = mean[rtype][key] + np.dot(step_exc_sample[key]*np.sqrt(eigval[rtype][key][:N_pc]),
                                                            eigvec[rtype][key][:,:N_pc].T)

                kstepexc_fwd = krate[:N_T]
                kstepexc_bkw = krate[N_T:]

                if is_log:
                    kstepexc_fwd = np.exp(kstepexc_fwd)
                    kstepexc_bkw = np.exp(kstepexc_bkw)



                fname = os.path.join(rate_dir, 'StepExcitation_' + key + '.h5')
                table[:,0] = T_Maxw
                table[:,1] = kstepexc_fwd
                with h5.File(fname, 'w') as f:
                    f.attrs['process'] = 'Ar(' + key.split('_')[0] + ') + e -> Ar(' + key.split('_')[1] +') + e'
                    f.attrs['temperature units'] = 'K'
                    f.attrs['rate units'] = 'm^3 / mol / s'
                    dset = f.create_dataset("table", table.shape, data=table)

                fname = os.path.join(rate_dir, 'StepExcitation_' + sfwd_to_bkw[key] + '.h5')
                table[:,0] = T_Maxw
                table[:,1] = kstepexc_bkw
                # breakpoint()
                with h5.File(fname, 'w') as f:
                    f.attrs['process'] = 'Ar(' + key.split('_')[1] + ') + e -> Ar(' + key.split('_')[0] +') + e'
                    f.attrs['temperature units'] = 'K'
                    f.attrs['rate units'] = 'm^3 / mol / s'
                    dset = f.create_dataset("table", table.shape, data=table)









# if not perform_sensitivity:
#     exit()





