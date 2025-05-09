import os
# import yaml
import configparser
import shutil
"""
Script to produce tps input files from reaction rate sample data

Options include a template input file and a directory containing the rate
samples in .h5 format, same as torch1d


"""


home = os.environ["HOME"]

# template_file = f"{home}/bedonian1/mean_r6/torch1d_input_r.yml"
# template_file = f"{home}/bedonian1/mean_tps2d_r6/lomach.torch.reacting.ini"
template_file = f"{home}/bedonian1/mean_tps2d_r6/r_lomach.torch.reacting.ini"
sample_dir = f"{home}/bedonian1/rate_mf_r1_pilot/"
restart_file = f"{home}/bedonian1/mean_tps2d_r6/restart_output-torch.sol.h5"
output_dir = f"{home}/bedonian1/tps2d_mf_r1_pilot_4/"



restart_fname = restart_file.split('/')[-1]
# don't make files for all 256 samples, start with the first few
# sample_start = 0
# sample_limit = 8
# sample_start = 8
# sample_limit = 16
# sample_start = 16
# sample_limit = 32
sample_start = 32
sample_limit = 48

dt = 1e-8
dt_next = 1e-6

formation_energy = {'Ar*': 1.114e6, # full lumped excited
                    'Ar.+1': 1520571.3883, # ionized
                    'Ar_m': 1116419.84847284, # metastable 4s
                    'Ar_r': 1129622.58232383, # resonant 4s
                    'Ar_p': 1267887.18783722, # 4p
                    'Ar_h': 1393459.40561185 # higher
}

name_to_species = {'meta':'Ar_m',
                   'res':'Ar_r',
                   'fourp':'Ar_p',
                   'higher':'Ar_h',
                   'Ground':'Ar'
                   }

name_to_st = {
    'meta':'1 0 0 0',
    'res':'0 1 0 0',
    'fourp':'0 0 1 0',
    'higher':'0 0 0 1'
}

#Equation name
def excitation_eq(name, name2):
    return 'Ar + E => ' + name_to_species[name] + ' + E'

def deexcitation_eq(name, name2):
    return name_to_species[name] + ' + E => Ar + E'

def ionization_eq(name, name2):
    return name_to_species[name] + ' + E => Ar.+1 + 2 E'

def recombination_eq(name, name2):
    return 'Ar.+1 + 2 E => ' + name_to_species[name] + ' + E'

def stepexcitation_eq(name, name2):
    return name_to_species[name] + ' + E => ' + name_to_species[name2] + ' + E'


#Stoichiometry
def ground_st():
    return '1 1 0 0 0 0 0'

def excited_st(name):
    return '0 1 ' + name_to_st[name] + ' 0'

def ion_st():
    return '0 2 0 0 0 0 1'

def listdir_nopickle(path):
    return [f for f in os.listdir(path) if not f.endswith('.pickle')]


reaction_eq_dict = {
    'Excitation': excitation_eq,
    'Deexcitation':deexcitation_eq,
    'Ionization':ionization_eq,
    'Recombination':recombination_eq,
    'StepExcitation':stepexcitation_eq,
}

######### Open Template

template = configparser.ConfigParser()
template.optionxform = str
with open(template_file, 'r') as f:
    template.read_file(f)
    # template = yaml.safe_load(f)

######### Create Output Dir
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

######### Loop through sample directories in sample_dir
samples = listdir_nopickle(sample_dir)
samples.sort()

if sample_limit:
    samples = samples[sample_start:sample_limit]
# breakpoint()
for sample in samples:

    rate_dir = sample_dir + '/' + sample + '/rates'

    # check if there is a directory called rates
    if not os.path.isdir(rate_dir):
        continue

    # create run directory for the sample
    sample_run_dir = output_dir + '/' + sample
    if not os.path.isdir(sample_run_dir):
        os.mkdir(sample_run_dir)

    # check rates
    rates = os.listdir(rate_dir)
    # if no4phigher:
    #     rates.remove('StepExcitation_fourp_higher.h5')
    #     rates.remove('StepExcitation_higher_fourp.h5')

    nSpecies = template['species']['numSpecies'] 

    # assume template contains properties for each species


    # loop over rate data again to add reactions
    # if template['reactions'] is None:
    #     template['reactions'] = []

    # re_list = [template['reactions'][i]['equation'] for i in range(len(template['reactions']))]


    # for rate in rates:

    #     rate_split = rate.split('.')[0].split('_')
    #     rtype = rate_split[0] # excitation, deexcitation, ionization, recombination, step-excitation, etc.
    #     r1 = rate_split[1]
    #     r2 = None
    #     if len(rate_split) == 3:
    #         r2 = rate_split[2]


    #     # check if reaction is accounted for
    #     reaction_name = reaction_eq_dict[rtype](r1, r2)
    #     if reaction_name not in re_list:
    #         template['reactions'].append({
    #             'equation': reaction_name,
    #             'rate_type': 'table',
    #             'table':{
    #                 'filename': '',
    #                 'x_logscale': False,
    #                 'y_logscale': True,
    #                 'interpolation': True
    #             }
    #         })

    template['time']['dt_fixed'] = f'{dt}'

    # list of reactions
    klist_f = list(template.keys())
    klist = []
    for key in klist_f:
        if 'reactions/' in key:
            klist.append(key)

    # finally, loop over rate data to set filenames
    for rate in rates:

        rate_split = rate.split('.')[0].split('_')
        rtype = rate_split[0] # excitation, deexcitation, ionization, recombination, step-excitation, etc.
        r1 = rate_split[1]
        r2 = None
        if len(rate_split) == 3:
            r2 = rate_split[2]

        reaction_name = "'" + reaction_eq_dict[rtype](r1, r2) + "'"

        # set filename
        # for item in template['reactions']:
        for item in klist:
            if template[item]['equation'] == reaction_name:
                template[item]['tabulated/filename'] = "'" + rate_dir + '/' + rate + "'"
                

    # set output file
    # template['io']['outdirBase'] = output_dir + '/' + sample + '/output-torch'

    # copy the restart file
    # DO NOT DO THIS, DO IT SEPARATELY!!!!
    # shutil.copy2(restart_file, output_dir + '/' + sample)

    # manage restarts, find most current restart file
    # template['io']['restartBase'] = output_dir + '/' + sample + '/' + restart_fname

    
    # write to file
    write_name = output_dir + '/' + sample + '/tps_axi2d_input.ini'
    with open(write_name, 'w') as f:
        template.write(f)

    # write a larger timestep version to file as well
    template['time']['dt_fixed'] = f'{dt_next}'
    write_name = output_dir + '/' + sample + '/tps_axi2d_input_LT.ini'
    with open(write_name, 'w') as f:
        template.write(f)

    
    