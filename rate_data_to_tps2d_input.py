import os
# import yaml
import configparser

"""
Script to produce tps input files from reaction rate sample data

Options include a template input file and a directory containing the rate
samples in .h5 format, same as torch1d


"""


home = os.environ["HOME"]

# template_file = f"{home}/bedonian1/mean_r6/torch1d_input_r.yml"
template_file = f"{home}/bedonian1/mean_tps2d_r6/tps2d_input_r.yml"
sample_dir = f"{home}/bedonian1/rate_resample_r7/"
output_dir = f"{home}/bedonian1/torch1d_resample_r7/"


formation_energy = {'Ar*': 1.114e6, # full lumped excited
                    'Ar.+1': 1.521e6, # ionized
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

def excitation_eq(name, name2):
    return 'Ar + E => ' + name_to_species[name] + ' + E'

def deexcitation_eq(name, name2):
    return name_to_species[name] + ' + E => Ar + E'

def ionization_eq(name, name2):
    return name_to_species[name] + ' + E => Ar.+1 + E + E'

def recombination_eq(name, name2):
    return 'Ar.+1 + E + E => ' + name_to_species[name] + ' + E'

def stepexcitation_eq(name, name2):
    return name_to_species[name] + ' + E => ' + name_to_species[name2] + ' + E'

reaction_eq_dict = {
    'Excitation': excitation_eq,
    'Deexcitation':deexcitation_eq,
    'Ionization':ionization_eq,
    'Recombination':recombination_eq,
    'StepExcitation':stepexcitation_eq,
}

######### Open Template

template = configparser.ConfigParser()
with open(template_file) as f:
    template.readfp(f)
    # template = yaml.safe_load(f)

######### Create Output Dir
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

######### Loop through sample directories in sample_dir
samples = os.listdir(sample_dir)
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
    if no4phigher:
        rates.remove('StepExcitation_fourp_higher.h5')
        rates.remove('StepExcitation_higher_fourp.h5')

    # loop over rate data available once 
    # first, check if the argon state is accounted for
    # NOTE: assuming template has E, Ar, Ar+ and no Ar* already
    for rate in rates:
        
        rate_split = rate.split('.')[0].split('_')
        rtype = rate_split[0] # excitation, deexcitation, ionization, recombination, step-excitation, etc.
        r1 = rate_split[1]
        r2 = None
        if len(rate_split) == 3:
            r2 = rate_split[2]

        sp_list = [template['species'][i]['name'] for i in range(len(template['species']))]
        
        if name_to_species[r1] not in sp_list:
            template['state']['species'].append(name_to_species[r1])
            template['species'].append({
                'name': name_to_species[r1],
                'composition': {'Ar': 1},
                'formation_energy': formation_energy[name_to_species[r1]]
            })

        if r2 is not None and name_to_species[r2] not in sp_list:
            template['state']['species'].append(name_to_species[r2])
            template['species'].append({
                'name': name_to_species[r2],
                'composition': {'Ar': 1},
                'formation_energy': formation_energy[name_to_species[r2]]
            })

    # second, loop over rate data again to add reactions
    if template['reactions'] is None:
        template['reactions'] = []

    re_list = [template['reactions'][i]['equation'] for i in range(len(template['reactions']))]

    for rate in rates:

        rate_split = rate.split('.')[0].split('_')
        rtype = rate_split[0] # excitation, deexcitation, ionization, recombination, step-excitation, etc.
        r1 = rate_split[1]
        r2 = None
        if len(rate_split) == 3:
            r2 = rate_split[2]


        # check if reaction is accounted for
        reaction_name = reaction_eq_dict[rtype](r1, r2)
        if reaction_name not in re_list:
            template['reactions'].append({
                'equation': reaction_name,
                'rate_type': 'table',
                'table':{
                    'filename': '',
                    'x_logscale': False,
                    'y_logscale': True,
                    'interpolation': True
                }
            })

    # finally, loop over rate data again to set filenames
    for rate in rates:

        rate_split = rate.split('.')[0].split('_')
        rtype = rate_split[0] # excitation, deexcitation, ionization, recombination, step-excitation, etc.
        r1 = rate_split[1]
        r2 = None
        if len(rate_split) == 3:
            r2 = rate_split[2]

        reaction_name = reaction_eq_dict[rtype](r1, r2)
        # set filename
        for item in template['reactions']:
            if item['equation'] == reaction_name:
                item['table']['filename'] = rate_dir + '/' + rate

    # set output file
    template['output']['directory'] = output_dir + '/' + sample + '/output'

    
    # write to file
    write_name = output_dir + '/' + sample + '/torch1d_input.yml'
    with open(write_name, 'w') as f:
        yaml.safe_dump(template, f)

    
