from scipy.stats.qmc import scale
import numpy as np
from sobol_tools import *

#------------------------------------------------------------------------------

"""
Class for managing sample inputs between categorized dictionaries and flattened arrays

"""
class SampleData():
    

    def __init__(self, categories, sizes, precomp=None):

        #NOTE: precomp dimensions should override sizes, or at least assert

        assert len(categories) == len(sizes)

        self.data = {}
        self.inds = {}

        # initialize and generate full array indices
        c  = 0
        for cat, size in categories, sizes:
            self.data[cat] = np.array([size, 0])
            self.inds[cat] = list(range(c, c+size))
            c = c+size


        # if we have precomputed data in a similar dict, try adding it now
        if precomp is isinstance(dict):
            self.addData(precomp)

        # if precomp is a numpy array
        if precomp is isinstance(np.ndarray):
            self.addDataArray(precomp)

#------------------------------------------------------------------------------

    def createData(self, N, method='sobol'):
        # generate samples internally
        ndim = self.getNDim()
        
        add = sobolSampleGen(ndim, N)
        
        self.addDataArray(add)
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
        assert add is isinstance(np.ndarray)

        for key, ind in self.inds:
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
                ndim += len(list(data[k][0]))
            return ndim
            
        # features in key entry
        else:
            return len(list(data[key][0]))

#------------------------------------------------------------------------------

    def genSobolData(self):
        """
        Given the current sample data as A, generate 
        """

        # SD_A is the current instance
        SD_A = self

        SD_B = SampleData()


        return SD_A, SD_B, SD_AB
    
#------------------------------------------------------------------------------