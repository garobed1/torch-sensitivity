import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import scipy.constants as spc
import pickle
from mpi4py import MPI

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

home = os.getenv('HOME')
kl_model_dir = 'results/cross_section_samples_r7/'
nom_dir = home + "/torch-sensitivity/trevilo-cases/torch_7sp_chem/nominal/rate-coefficients/"
res_dir = home + "/bedonian1/rate_resamples_r7/"


#### SAMPLING INPUTS ####
# Generate some artificial distributions for every considered cross section
# Nsamples = 8000
# Nsamples = 12000
Nsamples = 4

sample_react_dict = {
    "Excitation": True,
    "Ionization": True,
    "StepExcitation": True
}
# sample_exc = True
# sample_ion = True
# sample_step_exc = False

make_plots = False

reaction_types_fwd = ['Excitation', 'Ionization', 'StepExcitation']
lumped_rates = ["meta", "res", "fourp", "higher"]
lumped_rates_g = ["meta", "res", "fourp", "higher", "Ground"]
lumped_rates_fwd_step = ["meta_res", "meta_fourp", "meta_higher", 
                        "res_fourp", "res_higher", "fourp_higher"]
reaction_types_bwd = ['Excitation', 'Deexcitation', 'Ionization', 'Recombination']


# determine number of principal components/independent variables for each sample based on eigval threshold
pc_threshold = 1.0 - 1.e-8


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
# with open(kl_model_dir + "/frac.pickle", 'rb') as f:
#     frac = pickle.load(f)    
 
# determine number of vars per, should be given solely by pc_threshold
frac = {}
for rtype in reaction_types_fwd:
    frac[rtype] = {}
    lumped_rates_check = lumped_rates
    if rtype == "Ionization" or rtype == "Recombination":  
        lumped_rates_check = lumped_rates_g
    if rtype == "StepExcitation":  
        lumped_rates_check = lumped_rates_fwd_step

    for rate in lumped_rates_check:
        # print("{0} Indices: ".format(rate))
        frac[rtype][rate] = computeVarianceFractions(eigval[rtype][rate])
        # print(frac[rtype][rate])


N_pc = {}
for rtype in reaction_types_fwd:
    N_pc[rtype] = {}
    for rate in frac[rtype].keys():
        c = 0
        while frac[rtype][rate][c] < pc_threshold:
            c += 1

        N_pc[rtype][rate] = c + 0

print(N_pc)
breakpoint()












if rank == 0:
    if excluded_states:
        with open(sample_dir +'/excluded_states.pickle', 'wb') as f:
            pickle.dump(excluded_states, f)

    if os.path.exists(sample_dir + 'excluded_states.pickle'):
        with open(sample_dir + "excluded_states.pickle", 'rb') as f:
            excluded_states = pickle.load(f)
    else:
        excluded_states = []
excluded_states = comm.bcast(excluded_states, root=0)

# Constants
qe = spc.e    # 1.60217663e-19 [C]
kB = spc.k    # 1.380649e-23 [J/K]
NA = spc.N_A  # 6.022e23 [#/mol]
eV = qe/kB    # 11604.518 [K / eV]

# temp grid
T_Maxw = np.linspace(0.02, 2, 512) * eV

# lumped rates
lumped_rates = ["meta", "res", "fourp", "higher"]


# process considered configurations like in the chemistry to assign perturbations



# number of quantum states considered
Nconfig = len(total_config)

level_dict = {}

for i in range(len(total_config)):
    level_dict[total_config[i]] = i + 1

level_dict['Ground'] = 0


# total_config = []
# for i in range(0,len(df)):
#     cfg = configuration[i] + "-" + term[i] + "-" + str(J[i])
#     total_config.append(cfg)





# then keep track of "known" configs only
config_perturb_dist = {}
perturb_type = {}
# "Excitation" from ground state
# "Ionization" from ground and excited states
# one category for each StepExcitation original state i
Nvars = {
    # "Excitation": 0,
    # "Ionization": 0,    
    # "StepExcitation" + cfg: 0
}
sizes = {}


# for i in range(0,len(df)):
#     base_config = configuration[i][19:]
#     if base_config in known_configurations:

#NOTE: The bayesian samples, the first 7 or so may be way off, need to recompute
###############################################################################################
# Characterize uncertainties
for rtype in sample_react_dict.keys():

    if not sample_react_dict[rtype]:
        continue



    if rtype == "Excitation":
        Nvars[rtype] = 0
        
        # Nconfig
        sizes[rtype] = {}
        config_perturb_dist[rtype] = {}
        perturb_type[rtype] = {}

        #NOTE: one variable per config
        for cfg in total_config:
            # check if bayesian samples exists for this
            sfname = ''
            if bayesian_dirs is not None:
                sfname = bayesian_dirs[rtype] + 'crs.' + rtype.lower() + '.' + str(level_dict[cfg]) + '.dat.npy'
            if (sample_type == "bayesian"):
                if os.path.exists(sfname):
            
                    sarr = None
                    if rank == 0:
                        with open(sfname, 'rb') as f:
                            sarr = np.load(sfname)
                    sarr = comm.bcast(sarr, root=0)
                        
                    # NOTE: subtract the two matern parameters
                    Nvars[rtype] += sarr.shape[1] - 2
                    sizes[rtype][cfg] = sarr.shape[1] - 2
                    config_perturb_dist[rtype][cfg] = {"dist":"inferred", "datadir": sfname}
                    # perturb_type[rtype][cfg] = "inferred"
                else:
                    sfname = bayesian_dirs[rtype] + 'crs.' + rtype.lower() + '.' + str(level_dict[cfg]) + '.extrapolated.dat'
                    sarr = None
                    if rank == 0:
                        with open(sfname, 'rb') as f:
                            sarr = np.load(sfname)
                    sarr = comm.bcast(sarr, root=0)
                    # format var : [mean, rel_std]
                    #        var2: [mean, rel_std]

                    Nvars[rtype] += sarr.shape[0]
                    sizes[rtype][cfg] = sarr.shape[0]

                    # lower bound on vars
                    lbound = [0., 0.]
                    # note, assuming we have a relative standard deviation
                    config_perturb_dist[rtype][cfg] = {"dist":"normal", "model":True, "loc": list(sarr[:,0]), "scale": list(abs(sarr[:,1]*sarr[:,0])), "lbound":lbound}
            else:
                Nvars[rtype] += 1
                sizes[rtype][cfg] = 1
                config_perturb_dist[rtype][cfg] = {"dist":"normal", "model":False, "loc":0, 
                                                    "scale": logperturb}
    if rtype == "Ionization":
        Nvars[rtype] = 0
        # Nvars[rtype] = Nconfig + 1
        sizes[rtype] = {}
        config_perturb_dist[rtype] = {}

        for cfg in ['Ground'] + total_config:
            
            # check if bayesian samples exists for this
            sfname = ''
            if bayesian_dirs[rtype] is not None:
                sfname = bayesian_dirs[rtype] + 'crs.' + rtype.lower() + '.' + str(level_dict[cfg]) + '.dat.npy'
            if (sample_type == "bayesian"):
                if os.path.exists(sfname):
            
                    sarr = None
                    if rank == 0:
                        with open(sfname, 'rb') as f:
                            sarr = np.load(sfname)
                    sarr = comm.bcast(sarr, root=0)
                    # NOTE: subtract the two matern parameters
                    Nvars[rtype] += sarr.shape[1] - 2
                    sizes[rtype][cfg] = sarr.shape[1] - 2
                    config_perturb_dist[rtype][cfg] = {"dist":"inferred", "datadir": sfname}
                else:
                    sfname = bayesian_dirs[rtype] + 'crs.' + rtype.lower() + '.' + str(level_dict[cfg]) + '.extrapolated.dat'
                    sarr = None
                    if rank == 0:
                        with open(sfname, 'rb') as f:
                            sarr = np.load(sfname)
                    sarr = comm.bcast(sarr, root=0)
                    # format var : [mean, rel_std]
                    #        var2: [mean, rel_std]

                    Nvars[rtype] += sarr.shape[0]
                    sizes[rtype][cfg] = sarr.shape[0]

                    # assign bounds if needed

                    lbound = [1., 0., 0., 0.]


                    # note, assuming we have a relative standard deviation
                    config_perturb_dist[rtype][cfg] = {"dist":"normal", "model":True, "loc": list(sarr[:,0]), "scale": list(abs(sarr[:,1]*sarr[:,0])), "lbound":lbound}
    
            else: # "logperturb"
                Nvars[rtype] += 1
                sizes[rtype][cfg] = 1
                config_perturb_dist[rtype][cfg] = {"dist":"normal", "model":False, "loc":0, 
                                                    "scale": logperturb}
    if rtype == "StepExcitation":
        for cfg_i in total_config[:-1]:
            # StepExcNvars[rtype + cfg] = Nconfig - level_dict[cfg]
            Nvars[rtype + cfg_i] = 0
            
            # Nconfig - level_dict[cfg_i]

            sizes[rtype + cfg_i] = {}
            config_perturb_dist[rtype + cfg_i] = {}
            #NOTE: one variable per config
            for cfg_j in total_config[level_dict[cfg_i]:]:
                # check if bayesian samples exists for this
                # NOTE: Note this naming convention for inferred stepwise excitation sample files
                # e. g. crs.stepexcitation.3_29.dat
                sfname = ''
                if bayesian_dirs[rtype] is not None:
                    sfname = bayesian_dirs[rtype] + 'crs.' + rtype.lower() + '.' + str(level_dict[cfg_i]) + '_' + str(level_dict[cfg_j]) +'.dat.npy'
                if (sample_type == "bayesian"):
                    if os.path.exists(sfname):

                        sarr = None
                        if rank == 0:
                            with open(sfname, 'rb') as f:
                                sarr = np.load(sfname)
                        sarr = comm.bcast(sarr, root=0)
                        # NOTE: subtract the two matern parameters
                        Nvars[rtype + cfg_i] += sarr.shape[1] - 2
                        sizes[rtype + cfg_i][cfg_j] = sarr.shape[1] - 2
                        config_perturb_dist[rtype + cfg_i][cfg_j] = {"dist":"inferred", "datadir": sfname}
                    else:

                        sfname = bayesian_dirs[rtype] + 'crs.' + rtype.lower() + '.' + str(level_dict[cfg_i]) + '_' + str(level_dict[cfg_j]) +'.extrapolated.dat'
                        sarr = None
                        if rank == 0:
                            with open(sfname, 'rb') as f:
                                sarr = np.load(sfname)
                        sarr = comm.bcast(sarr, root=0)
                        # format var : [mean, rel_std]
                        #        var2: [mean, rel_std]

                        Nvars[rtype + cfg_i] += sarr.shape[0]
                        sizes[rtype + cfg_i][cfg_j] = sarr.shape[0]

                        # lower bound on vars
                        lbound = [0., 0.]
                        
                        # note, assuming we have a relative standard deviation
                        config_perturb_dist[rtype + cfg_i][cfg_j] = {"dist":"normal", "model":True, "loc": list(sarr[:,0]), "scale": list(abs(sarr[:,1]*sarr[:,0])), "lbound":lbound}

                            
                else: # "logperturb"
                    Nvars[rtype + cfg_i] += 1
                    sizes[rtype + cfg_i][cfg_j] = 1
                    config_perturb_dist[rtype + cfg_i][cfg_j] = {"dist":"normal", "model":False, "loc":0, 
                                                    "scale": logperturb}
            # Nvars[rtype] += StepExcNvars[cfg]

    


    # or, get samples from bayesian inference of experimental data (Chung)
    # obtained by running bayesian_cross_section_ext_*.py
    # NOTE: Make sure to use the correct model!
    # if sample_type == 'bayesian':

    #     if sample_exc:

    #         Nvars += Nvars_exc



NvarsTotal = 0
for key, item in Nvars.items():
    NvarsTotal += item



# mapABReaction(Nsamples*Nvars_exc + Nsamples*Nvars_ion + 89, Nsamples, 'AB', True, True, True, Nvars_exc, Nvars_ion, Nvars_step_exc)
# breakpoint()




################################################################################################
# Sample Generation
# Ground Excitation

perturb_samples_base = {}
perturb_samples_A = {}
perturb_samples_B = {}
perturb_samples_AB = {}

for rtype in config_perturb_dist.keys():

# if sample_exc:
    perturb_samples_base_r0 = None
    if rank == 0:
        perturb_samples_base_r0 = SampleData(categories=list(config_perturb_dist[rtype].keys()),
                                        sizes=sizes[rtype], scales=config_perturb_dist[rtype])
        perturb_samples_base_r0.createData(N=Nsamples)
    
    perturb_samples_base[rtype] = comm.bcast(perturb_samples_base_r0, root=0)

    if "StepExcitation" in rtype:
        a_sample_file = sample_dir + rtype[:14] + str(level_dict[rtype[14:]]) + "_perturb_samples_A.pickle"
        b_sample_file = sample_dir + rtype[:14] + str(level_dict[rtype[14:]]) + "_perturb_samples_B.pickle"
    else:
        a_sample_file = sample_dir + rtype + "_perturb_samples_A.pickle"
        b_sample_file = sample_dir + rtype + "_perturb_samples_B.pickle"

    perturb_samples_A[rtype] = None
    perturb_samples_B[rtype] = None
    perturb_samples_AB[rtype] = None
    if rank == 0:
        if os.path.exists(a_sample_file):
            if compute_sobol:
                # perturb_samples_A[rtype] = None
                # perturb_samples_B[rtype] = None
                # perturb_samples_AB[rtype] = None
                # if rank == 0:
                with open(a_sample_file, 'rb') as f:
                    perturb_samples_A[rtype] = pickle.load(f)
                with open(b_sample_file, 'rb') as f:
                    perturb_samples_B[rtype] = pickle.load(f)
                perturb_samples_A[rtype], perturb_samples_B[rtype], perturb_samples_AB[rtype] = perturb_samples_A[rtype].genSobolData(perturb_samples_B[rtype])

                # perturb_samples_A[rtype] = comm.bcast(perturb_samples_A[rtype], root=0)
                # perturb_samples_B[rtype] = comm.bcast(perturb_samples_B[rtype], root=0)
                # perturb_samples_AB[rtype] = comm.bcast(perturb_samples_AB[rtype], root=0)

            else:
                # perturb_samples_A[rtype] = None
                # perturb_samples_B[rtype] = None
                # perturb_samples_AB[rtype] = None
                # if rank == 0:
                with open(a_sample_file, 'rb') as f:
                    perturb_samples_A[rtype] = pickle.load(f)   
                # if "StepExcitation" in rtype:
                #     breakpoint()
                # perturb_samples_A[rtype] = comm.bcast(perturb_samples_A[rtype], root=0)

        else:
            if compute_sobol:
                # perturb_samples_A_r0 = None 
                # perturb_samples_B_r0 = None
                # perturb_samples_AB_r0 = None
                # if rank == 0:
                perturb_samples_A_r0, perturb_samples_B_r0, perturb_samples_AB_r0 = perturb_samples_base[rtype].genSobolData()
                # perturb_samples_A[rtype] = comm.bcast(perturb_samples_A_r0, root=0)
                # perturb_samples_B[rtype] = comm.bcast(perturb_samples_B_r0, root=0)


                perturb_samples_A[rtype], perturb_samples_B[rtype], perturb_samples_AB[rtype] = perturb_samples_base[rtype].genSobolData(perturb_samples_B[rtype])

                # if rank == 0:
                with open(a_sample_file, 'wb') as f:
                    pickle.dump(perturb_samples_A[rtype], f)
                with open(b_sample_file, 'wb') as f:
                    pickle.dump(perturb_samples_B[rtype], f)
                    # pickle(exc_perturb_samples_A)
                # comm.Barrier()
            else:
                perturb_samples_A[rtype] = perturb_samples_base[rtype]
                # if rank == 0:
                with open(a_sample_file, 'wb') as f:
                    pickle.dump(perturb_samples_A[rtype], f)
                # comm.Barrier()
    perturb_samples_A[rtype] = comm.bcast(perturb_samples_A[rtype], root=0)
    perturb_samples_B[rtype] = comm.bcast(perturb_samples_B[rtype], root=0)
    perturb_samples_AB[rtype] = comm.bcast(perturb_samples_AB[rtype], root=0)

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

### TEST
# groups = ['AB']

sample_exc = False
sample_ion = False
sample_step_exc = False
if 'Excitation' in config_perturb_dist.keys():
    sample_exc = True
if 'Ionization' in config_perturb_dist.keys():
    sample_ion = True
if 'StepExcitation' + total_config[0] in config_perturb_dist.keys():
    sample_step_exc = True

# potential speedup
exc_preload = None
if rank == 0:
    if os.path.exists(sample_dir + '/argon_excitation_sigma.pickle'):
        with open(sample_dir + '/argon_excitation_sigma.pickle', 'rb') as f:
            exc_preload = pickle.load(f)
exc_preload = comm.bcast(exc_preload, root=0)

# potential speedup
ion_preload = None
if rank == 0:
    if os.path.exists(sample_dir + '/argon_ionization_sigma.pickle'):
        with open(sample_dir + '/argon_ionization_sigma.pickle', 'rb') as f:
            ion_preload = pickle.load(f)
ion_preload = comm.bcast(ion_preload, root=0)

# potential speedup
step_exc_preload = None
if rank == 0:
    if os.path.exists(sample_dir + '/argon_stepwise_excitation_sigma.pickle'):
        with open(sample_dir + '/argon_stepwise_excitation_sigma.pickle', 'rb') as f:
            step_exc_preload = pickle.load(f)
step_exc_preload = comm.bcast(step_exc_preload, root=0)

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
        results_dir = "{0}sig_{1}_{2:06d}/plots".format(sample_dir, group, isamp)


        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        # Place to write data
        rate_dir = "{0}sig_{1}_{2:06d}/rates".format(sample_dir, group, isamp)
        if not os.path.isdir(rate_dir):
            os.makedirs(rate_dir)
        
        # Avoid recomputing if we can
        # if os.path.exists(results_dir):
        if checkRateSample(rate_dir, sample_exc, sample_ion, sample_step_exc):
            continue
        
        group_exc = group
        group_ion = group
        group_step_exc = group

        # map out appropriate samples from each reaction group
        ind_samp_dict, group_dict = mapABReaction2(isamp, Nsamples, group, Nvars, NvarsTotal)
        # ind_samp_exc, ind_samp_ion, ind_samp_step_exc, group_exc, group_ion, group_step_exc, key_AB_ind  = mapABReaction(isamp, Nsamples, group, 
        #                     sample_exc, sample_ion, sample_step_exc, 
        #                     Nvars['Excitation'], Nvars['Ionization'], Nvars['StepExcitation'])


        ############################################ TODO ##################################################

                                            # Left off here, continue 
                                            
                                            # finish this file, partic organizing step excitation samples

        ############################################ TODO ##################################################


        # Evaluate the lumped electron-impact process rates...
        if sample_exc:

            
            

            # assemble perturbations
            exc_sample = perturb_samples["Excitation"][group_dict['Excitation']](ind_samp_dict['Excitation'])
            exc_type = {}
            for key in config_perturb_dist["Excitation"].keys():
                if config_perturb_dist["Excitation"][key]['dist'] != 'inferred' and config_perturb_dist["Excitation"][key]['model'] == False:
                    exc_type[key] = 'log_constant'
                elif config_perturb_dist["Excitation"][key]['dist'] != 'inferred' and config_perturb_dist["Excitation"][key]['model'] == True:
                    exc_type[key] = model_dict["Excitation"]
                else:
                    exc_type[key] = model_dict['Excitation']
            perturb_exc = {'type':exc_type, 'data':exc_sample}

            TK, kexcite_fwd, kexcite_bkw = excitation.evaluate_excitation_rates(df, results_dir, sig_perturb=perturb_exc, sig_preload=exc_preload, excluded_states=excluded_states, make_plots=make_plots)

            # Excitation rates
            table = np.zeros((len(TK), 2))
            for key in kexcite_fwd.keys():
                fname = os.path.join(rate_dir, 'Excitation_' + key + '.h5')
                table[:,0] = TK
                table[:,1] = kexcite_fwd[key] * spc.N_A
                with h5.File(fname, 'w') as f:
                    f.attrs['process'] = 'Ar(Ground) + e -> Ar('+key+') + e'
                    f.attrs['temperature units'] = 'K'
                    f.attrs['rate units'] = 'm^3 / mol / s'
                    dset = f.create_dataset("table", table.shape, data=table)

            # Deexcitation rates
            table = np.zeros((len(TK), 2))
            for key in kexcite_bkw.keys():
                fname = os.path.join(rate_dir, 'Deexcitation_' + key + '.h5')
                table[:,0] = TK
                table[:,1] = kexcite_bkw[key] * spc.N_A
                with h5.File(fname, 'w') as f:
                    f.attrs['process'] = 'Ar('+key+') + e -> Ar(Ground) + e'
                    f.attrs['temperature units'] = 'K'
                    f.attrs['rate units'] = 'm^3 / mol / s'
                    dset = f.create_dataset("table", table.shape, data=table)
        
        # # 2) Ionization
        if sample_ion:

            
            
            # assemble perturbations
            
            ion_sample = perturb_samples["Ionization"][group_dict['Ionization']](ind_samp_dict['Ionization'])
            ion_type = {}
            for key in config_perturb_dist["Ionization"].keys():
                if config_perturb_dist["Ionization"][key]['dist'] != 'inferred' and config_perturb_dist["Ionization"][key]['model'] == False:
                    ion_type[key] = 'log_constant'
                elif config_perturb_dist["Ionization"][key]['dist'] != 'inferred' and config_perturb_dist["Ionization"][key]['model'] == True:
                    ion_type[key] = model_dict["Ionization"]
                else:
                    ion_type[key] = model_dict["Ionization"]
            perturb_ion = {'type':ion_type, 'data':ion_sample}
            
            TK, kion_fwd, kion_bkw = ionization.evaluate_ionization_rates(df, results_dir, sig_perturb=perturb_ion, sig_preload=ion_preload, excluded_states=excluded_states, make_plots=make_plots)

            # Forward ionization rates
            table = np.zeros((len(TK), 2))
            for key in kion_fwd.keys():
                fname = os.path.join(rate_dir, 'Ionization_' + key + '.h5')
                table[:,0] = TK
                table[:,1] = kion_fwd[key] * spc.N_A
                with h5.File(fname, 'w') as f:
                    f.attrs['process'] = 'Ar('+key+') + e -> Ar+ + e + e'
                    f.attrs['temperature units'] = 'K'
                    f.attrs['rate units'] = 'm^3 / mol / s'
                    dset = f.create_dataset("table", table.shape, data=table)

            # Reverse ionization (recombination) rates
            table = np.zeros((len(TK), 2))
            for key in kion_bkw.keys():
                fname = os.path.join(rate_dir, 'Recombination_' + key + '.h5')
                table[:,0] = TK
                table[:,1] = kion_bkw[key] * spc.N_A * spc.N_A
                with h5.File(fname, 'w') as f:
                    f.attrs['process'] = 'Ar+ + e + e -> Ar('+key+') + e'
                    f.attrs['temperature units'] = 'K'
                    f.attrs['rate units'] = 'm^6 / mol^2 / s'
                    dset = f.create_dataset("table", table.shape, data=table)


        # # 3) Stepwise Excitation
        if sample_step_exc:
            
            # key_AB = None
            # if key_AB_ind is not None:
            #     key_AB = list(config_perturb_dist.keys())[key_AB_ind]

            # step_exc_samp = {}
            # for ikey in list(config_perturb_dist.keys())[:-1]:
            #     if ikey == key_AB:
            #         step_exc_samp[ikey] = step_exc_perturb_samples_AB[ikey](ind_samp_step_exc)
            #     else:
            #         step_exc_samp[ikey] = step_exc_perturb_samples_A[ikey](ind_samp_step_exc%Nsamples)

            # step_exc_perturb_samples[group_step_exc](ind_samp)
            # sig_perturb_step_exc = {}
            # for ikey in list(config_perturb_dist.keys())[:-1]:
            #     sig_perturb_step_exc[ikey] = {'type':'log_constant', 'data':step_exc_samp[ikey]}

            

            # assemble_perturbations
            step_exc_sample = {}
            step_exc_type = {}
            perturb_step_exc = {}
            for _ikey in config_perturb_dist.keys():
                if _ikey == "Excitation" or _ikey == "Ionization":
                    continue

                ikey = _ikey[14:]
                
                step_exc_sample[ikey] = perturb_samples[_ikey][group_dict[_ikey]](ind_samp_dict[_ikey])
                step_exc_type[ikey] = {}
                for jkey in config_perturb_dist[_ikey].keys():
                    if config_perturb_dist[_ikey][jkey]['dist'] != 'inferred' and config_perturb_dist[_ikey][jkey]['model'] == False:
                        step_exc_type[ikey][jkey]  = 'log_constant'
                    elif config_perturb_dist[_ikey][jkey]['dist'] != 'inferred' and config_perturb_dist[_ikey][jkey]['model'] == True:
                        step_exc_type[ikey][jkey] = model_dict["StepExcitation"]
                    else:
                        step_exc_type[ikey][jkey] = model_dict['StepExcitation']
                perturb_step_exc[ikey] = {'type':step_exc_type[ikey], 'data':step_exc_sample[ikey]}
            # breakpoint()
            TK, kstepwise = stepwise_excitation.evaluate_excitation_rates(df, results_dir, sig_perturb=perturb_step_exc, sig_preload=step_exc_preload, excluded_states=excluded_states, make_plots=make_plots)

            table = np.zeros((len(TK), 2))
            for ikey in kstepwise.keys():
                for jkey in kstepwise[ikey].keys():
                    if ikey == jkey:
                        continue

                    fname = os.path.join(rate_dir, 'StepExcitation_' + ikey + '_' + jkey + '.h5')
                    table[:,0] = TK
                    table[:,1] = kstepwise[ikey][jkey] * spc.N_A
                    with h5.File(fname, 'w') as f:
                        f.attrs['process'] = 'Ar(' + ikey + ') + e -> Ar(' + jkey +') + e'
                        f.attrs['temperature units'] = 'K'
                        f.attrs['rate units'] = 'm^3 / mol / s'
                        dset = f.create_dataset("table", table.shape, data=table)







# if not perform_sensitivity:
#     exit()





