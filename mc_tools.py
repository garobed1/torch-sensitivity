
import numpy as np


"""
Tools for generating monte carlo samples
"""

#------------------------------------------------------------------------------

def mcSampleGen(ndim, N):

    gen = np.random.default_rng()

    return gen.random([ndim, N])
    # return gen.random_base2(N)

#------------------------------------------------------------------------------


# if __name__ == "__main__":

#     from sample_utils import *
#     from scipy.stats import uniform

#     def func(x, y):
#         return 3*x + 10*y
    
#     M = 6
#     N = 2**M

#     in_cats = ['x', 'y']
#     in_sizes = [1, 1]
#     ins = SampleData(in_cats, in_sizes)
#     ins.createData(N)

#     # create sobol input sets
#     in_A, in_B, in_AB = ins.genSobolData()

#     res_A = np.zeros(N)
#     res_B = np.zeros(N)
#     for i in range(N):
#         res_A[i] = func(in_A(i, 'x')[0], in_A(i, 'y')[0])
#         res_B[i] = func(in_B(i, 'x')[0], in_B(i, 'y')[0])

#     # should be a dict
#     work = in_AB(list(range(0, N*2)))
#     res_AB = func(work['x'], work['y'])

#     out_A = SampleData(['r'], [1], np.atleast_2d(res_A))
#     out_B = SampleData(['r'], [1], np.atleast_2d(res_B))
#     out_AB = SampleData(['r'], [1], np.atleast_2d(res_AB))

#     # compute indices
#     r1, r2 = computeSobolIndices(out_A, out_B, out_AB, 2, 'r')

#     print(r1)
#     print(r2)

#     func2 = lambda x: func(x[0], x[1])
#     print(sobol_indices(func=func2, n=N, dists=[uniform(), uniform()]).first_order)

#     breakpoint()


