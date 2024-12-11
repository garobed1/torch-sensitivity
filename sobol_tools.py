from scipy.stats import sobol_indices
from scipy.stats.qmc import Sobol
import numpy as np





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

    #TODO FIX: arrays are modified in place by sobol_indices, they are centered
    # also, results are inconsistent
    func = {'f_A': f_A, 'f_B': f_B, 'f_AB': f_AB_n}
    res = sobol_indices(func=func, n=N_A)

    return res.first_order, res.total_order







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