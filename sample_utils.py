from scipy.stats.qmc import scale
from scipy.stats import norm, uniform, beta, lognorm
import numpy as np
from sobol_tools import *



dist_map = {
    "normal": norm,
    "uniform": uniform,
    "beta": beta,
    "lognormal"; lognorm
}

#------------------------------------------------------------------------------
"""
Class for managing sample data between categorized dictionaries and flattened arrays

Intended to be used for either inputs or outputs
"""
class SampleData():
    

    def __init__(self, categories, sizes, scales=None, precomp=None):

        #NOTE: precomp dimensions should override sizes, or at least assert
        assert len(categories) == len(sizes)

        self.data = {}
        self.inds = {}
        self.inds_inv = {}
        self.scales = {}

        # initialize and generate full array indices
        c  = 0
        for cat, size in zip(categories, sizes):
            self.data[cat] = np.empty([size, 0])
            self.inds[cat] = list(range(c, c+size))
            
            #NOTE: inds dictionary entries should be unique, so inverting should work
            c2 = 0
            for ind in self.inds[cat]:
                self.inds_inv[ind] = [cat, c2]
                c2+=1

            c = c+size


        # if we have precomputed data in a similar dict, try adding it now
        if isinstance(precomp, dict):
            self.addData(precomp)

        # if precomp is a numpy array
        if isinstance(precomp, np.ndarray):
            self.addDataArray(precomp)

        # possible references to either inputs that generated this data, or outputs generated from this data
        self.input_ref = None
        self.output_ref = None


#------------------------------------------------------------------------------

    def __call__(self, i, cat=None):

        #return sample i, which may be a list
        
        # return full dict if category unspecified
        if cat is None:    
            sample = {}

            for key, item in self.data:
                sample[key] = self.data[key][:, i]

            return sample
        
        # otherwise return array
        else:
            return self.data[cat][:,i]
#------------------------------------------------------------------------------

    def createData(self, N, scale=None, method='sobol'):
        # generate samples internally
        ndim = self.getNDim()
        
        #TODO: Implement other methods
        add = sobolSampleGen(ndim, N).T

        # scale is dict
        if scale is not None and self.scales is None:
            self.scales = scale
            add_scale = uniformToDist(add, self.scales, self.inds)
        elif self.scales is not None:
            if scale is not None:
                print("Distribution mapping already present")

            add_scale = uniformToDist(add, self.scales, self.inds)
            

        # uniform hypercube
        else:
            add_scale = add

        self.addDataArray(add_scale)
        
#------------------------------------------------------------------------------

    def addData(self, add):
        # essentially just copying a dict over and appending it
        assert add is isinstance(dict)

        # NOTE: it is possible to not include samples from certain categories at the moment
        # assert all categories have same number of samples
        add_cats = list(add.keys())
        n_add = self.data[add_cats[0]].shape[1]
        for key in add_cats:
            assert self.data[key].shape[1] == n_add
            assert self.data[key].shape[0] == self.getNDim(key)

        for key in add_cats:
            self.data[key] = np.append(self.data[key], add[key], axis=1)


        not_add = [i for i in list(self.data.keys()) if i not in add_cats ]
        for key in not_add:
            self.data[key] = np.append(self.data[key], np.empty([self.getNDim(key), n_add])*np.nan, axis=1)


#------------------------------------------------------------------------------

    def addDataArray(self, add):
        # NOTE: added array indexing should match self.inds, this is a riskier approach
        assert isinstance(add, np.ndarray)

        for key, ind in self.inds.items():
            self.data[key] = np.append(self.data[key], add[ind, :], axis=1)

#------------------------------------------------------------------------------

    def getInputArray(self):
        """
        Flatten samples to a 2D array according to the indexing of self.inds
        """
        ndim = self.getNDim()
        N = self.getNSamples()

        data_array = np.zeros([ndim, N])

        for key in self.getCategories():
            data_array[self.inds[key], :] = self.data[key][:,:]

        return data_array

#------------------------------------------------------------------------------

    def getNSamples(self):
        # get current number of samples
        cats = list(self.data.keys())
        return self.data[cats[0]].shape[1]

#------------------------------------------------------------------------------

    def getCategories(self):
        # get list of variable categories

        return list(self.data.keys())

#------------------------------------------------------------------------------

    def getNDim(self, key = None):
        # get number of features

        data = self.data

        # total features
        if key is None or key not in data.keys():

            ndim = 0
            for k in data.keys():
                ndim += data[k].shape[0]
            return ndim
            
        # features in key entry
        else:
            return data[key].shape[0]

#------------------------------------------------------------------------------

    def genSobolData(self):
        """
        Given the current sample data as A, generate 
        """
        cats = self.getCategories()
        sizes = [self.getNDim(x) for x in cats]

        # SD_A is the current instance
        SD_A = self

        # independent sample for B
        SD_B = SampleData(cats, sizes, precomp=sobolSampleGen(self.getNDim(), self.getNSamples()).T)
        # now produce SD_AB as a lambda function with __call__ similar to SampleData
        SD_AB = lambda x, cat=None : _sobolABGen(x, SD_A, SD_B, cat)
        # SD_AB = ABSamples(SD_A, SD_B)


        return SD_A, SD_B, SD_AB
    


# #------------------------------------------------------------------------------

# """
# Class that specifically manages the AB samples of a Sobol sensitivity analysis

# """
# class ABSamples(SampleData):

#     def __init__(self, SD_A, SD_B):
#         self.SD_A = SD_A
#         self.SD_B = SD_B

#         # all we need is to set up the outputs, the inputs are handled in __call__




#     def __call__(self, i, cat=None):

#         #return sample i of the AB sample set, which may be a list
#         return _sobolABGen(i, self.SD_A, self.SD_B, cat=cat)

#------------------------------------------------------------------------------


def _sobolABGen(i, SD_A, SD_B, cat):
    # helper for generating the AB samples

    N = SD_A.getNSamples()

    # grabbing B based on standard AB order
    A_ind = [j % N for j in list(i)]
    B_ind = [j // N for j in list(i)]

    #return sample i, which may be a list
        
    sample = {}

    for key in SD_A.data:
        sample[key] = SD_A.data[key][:, A_ind]

    # replace B entries
    B = SD_B.getInputArray()

    #TODO: finish this up
    for k in range(len(A_ind)):
    # for k in A_ind:
        cat_inv, ind_inv = SD_A.inds_inv[B_ind[k]]
        sample[cat_inv][ind_inv][k] = SD_B.data[cat_inv][ind_inv][A_ind[k]]

    
    # return full dict if category unspecified
    if cat is None:
        return sample
    # otherwise return array
    else:
        return sample[cat]
    

# use ppf to transform low discrepancy sequences like sobol to distributions
def uniformToDist(data, scales, inds):

    assert list(scales.keys()) == list(inds.keys())

    data_s = np.zeros_like(data)

    # iterate over indices
    for key, ind, in inds:
        scale_info = scales[key]

        if isinstance(scale_info["loc"], list):
            for i in range(len(ind)):

                data_s[ind[i], :] = transformDist(data_s[ind[i], :], 
                                scale_info['dist'][i], scale_info['loc'][i], scale_info['scale'][i])

        else: # apply same dist to all variables in category

            for i in range(len(ind)):

                data_s[ind[i], :] = transformDist(data_s[ind[i], :], 
                                scale_info['dist'], scale_info['loc'], scale_info['scale'])
                
    return data_s



# NOTE: no beta dist capabilities yet
def transformDist(x, dist, loc, scale, a=None, b=None):

    if a is not None and b is not None:
        x_s = dist_map[dist].ppf(x, loc=loc, scale=scale, a=a, b=b)
    else:
        x_s = dist_map[dist].ppf(x, loc=loc, scale=scale)

    return x_s