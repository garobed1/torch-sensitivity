import os, yaml

"""
Writes torch1d input file when restarting, 
"""
def t1dRestart(sdir, sfstep, infile):
    
    s_run_torch1d = True

    for fname in os.listdir(sdir + '/output/'):
        if fname.endswith(f'-{sfstep:08d}.h5'): 
            s_run_torch1d = False
            break

    # now check where the calculation stopped
    if s_run_torch1d and len(os.listdir(sdir + '/output/')):
        with open(sdir + '/' + infile) as f:
            torch1d_in = yaml.safe_load(f)

        ic = torch1d_in['state']['initial_condition']
        prefix = torch1d_in['prefix']

        cstep = 0
        fname_s = prefix + '-00000000.h5'
        for fname in os.listdir(sdir + '/output/'):
            if not fname.endswith('crashed.h5'):
                cand = int(fname[-11:-3])
                if cand > cstep:
                    cstep = cand
                    fname_s = fname

        # need to adjust the input file
        fstep_s = sfstep - cstep
        ic_s = sdir + '/output/' + fname_s

        torch1d_in['time_integration']['number_of_timesteps'] = fstep_s
        torch1d_in['state']['initial_condition'] = ic_s

        infile_r = 'r_' + infile
        with open(sdir + '/' + infile_r, 'w') as f:
            yaml.safe_dump(torch1d_in, f)

    else:
        infile_r = infile
    

    return infile_r, s_run_torch1d