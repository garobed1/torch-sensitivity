from scipy.stats import sobol_indices
from scipy.stats.qmc import Sobol
import numpy as np


"""
Tools for generating Sobol samples and computing variance-based sensitivity
analysis via Sobol indices
"""

#------------------------------------------------------------------------------

def sobolSampleGen(ndim, N):
    # Generate Sobol sequence samples

    gen = Sobol(ndim)

    return gen.random(N)
    # return gen.random_base2(N)

#------------------------------------------------------------------------------

def computeSobolIndices(O_A, O_B, O_AB, ndim, cat):
    # compute sobol indices from data produced from A, B, and AB sample sets
    # ndim is input dimensionality

    # assert list(O_A.data.keys()) == list(O_B.data.keys()) == list(O_AB.data.keys())
    assert O_A.data[cat].shape == O_A.data[cat].shape
    assert O_AB.data[cat].shape[0] == O_A.data[cat].shape[0]
    assert O_AB.data[cat].shape[1] == ndim*O_A.data[cat].shape[1]

    odim = O_A.data[cat].shape[0]
    N_A = O_A.data[cat].shape[1]

    f_A = np.copy(O_A.data[cat])
    f_B = np.copy(O_B.data[cat])
    f_AB = np.copy(O_AB.data[cat])
    f_AB_n = np.reshape(f_AB, [ndim, odim, N_A])

    func = {'f_A': f_A, 'f_B': f_B, 'f_AB': f_AB_n}
    res = sobol_indices(func=func, n=N_A)

    return res.first_order, res.total_order

# TODO: Write test for this function
# map AB sample indices to correct sample sets for plasma reaction types
def mapABReaction(isamp, Nsamples, group, sample_exc, sample_ion, sample_step_exc, 
                  Nvars_exc, Nvars_ion, Nvars_step_exc):

    if group == 'A' or group == 'B':
        return isamp%Nsamples, isamp%Nsamples, isamp%Nsamples, group, group, group, None
    
    # otherwise, we need to compute the appropriate AB index here
    reactions = []
    Nvars = 0
    if sample_exc:
        reactions.append(0)
        Nvars += Nvars_exc
    if sample_ion:
        reactions.append(1)
        Nvars += Nvars_ion
    if sample_step_exc:
        reactions.append(2)
        Nvars += Nvars_step_exc

    # first determine which reaction we're working with based on sample flags
    if len(reactions) == 0:
        print("No valid reactions!")
        return
    elif len(reactions) == 1:
        reaction_ind = 0
    elif len(reactions) == 2:
        if isamp < Nsamples*Nvars_exc:
            group_ion = 'A'
            group_exc = 'AB'
            group_step_exc = 'A'
            reaction_ind = 0
        else:
            group_exc = 'A'
            group_ion = 'AB'
            group_step_exc = 'A'
            reaction_ind = 1
    else:
        # replace excitation sample with B
        if isamp < Nsamples*Nvars_exc: 
            group_ion = 'A'
            group_step_exc = 'A'
            group_exc = 'AB'
            reaction_ind = 0

        # replace ionization sample with B
        elif isamp >= Nsamples*Nvars_exc and isamp < Nsamples*Nvars_exc + Nsamples*Nvars_ion:
            group_exc = 'A'
            group_step_exc = 'A'
            group_ion = 'AB'
            reaction_ind = 1

        # replace stepwise excitation sample with B
        else:
            group_exc = 'A'
            group_ion = 'A'
            group_step_exc = 'AB'
            reaction_ind = 2

    reaction_bin = reactions[reaction_ind]
       
    # NOTE: Currently only works if all three reaction types are considered for UQ
    # next determine the appropriate index
    # breakpoint()
    if reaction_bin == 0 or reaction_bin == 1:
        if reaction_ind == 0:
            breakpoint()

            return isamp, isamp%Nsamples, isamp%Nsamples, group_exc, group_ion, group_step_exc, None
        if reaction_ind == 1:
            breakpoint()
            return (isamp - Nsamples*Nvars_exc)%Nsamples, isamp - Nsamples*Nvars_exc, (isamp - Nsamples*Nvars_exc)%Nsamples, group_exc, group_ion, group_step_exc, None
    if reaction_bin == 2:
        isamp_base = isamp - Nsamples*Nvars_exc - Nsamples*Nvars_ion

        isamp_cand = 0 + isamp_base
        # transform to appropriate sub reaction index
        thresh = 0
        red = 0
        count = 0
        while isamp_base >= thresh:
            isamp_cand -= red
            red = (Nvars_exc - 1 - count)*Nsamples
            thresh += red
            count += 1

        return isamp_base%Nsamples, isamp_base%Nsamples, isamp_cand, group_exc, group_ion, group_step_exc, count-1




if __name__ == "__main__":

    from sample_utils import *
    from scipy.stats import uniform

    def func(x, y):
        return 3*x + 10*y
    
    M = 6
    N = 2**M

    in_cats = ['x', 'y']
    in_sizes = [1, 1]
    ins = SampleData(in_cats, in_sizes)
    ins.createData(N)

    # create sobol input sets
    in_A, in_B, in_AB = ins.genSobolData()

    res_A = np.zeros(N)
    res_B = np.zeros(N)
    for i in range(N):
        res_A[i] = func(in_A(i, 'x')[0], in_A(i, 'y')[0])
        res_B[i] = func(in_B(i, 'x')[0], in_B(i, 'y')[0])

    # should be a dict
    work = in_AB(list(range(0, N*2)))
    res_AB = func(work['x'], work['y'])

    out_A = SampleData(['r'], [1], np.atleast_2d(res_A))
    out_B = SampleData(['r'], [1], np.atleast_2d(res_B))
    out_AB = SampleData(['r'], [1], np.atleast_2d(res_AB))

    # compute indices
    r1, r2 = computeSobolIndices(out_A, out_B, out_AB, 2, 'r')

    print(r1)
    print(r2)

    func2 = lambda x: func(x[0], x[1])
    print(sobol_indices(func=func2, n=N, dists=[uniform(), uniform()]).first_order)

    breakpoint()


