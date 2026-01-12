from scipy.stats.qmc import scale
from scipy.stats import norm, uniform, beta, lognorm, truncnorm, foldnorm
import numpy as np
from util_tuq.sobol_tools import *
from util_tuq.mc_tools import *



dist_map = {
    "normal": norm,
    "uniform": uniform,
    "beta": beta,
    "lognormal": lognorm,
    "tnormal": truncnorm,
    "fnormal": foldnorm
}

#------------------------------------------------------------------------------
"""
Class for managing sample data between categorized dictionaries and flattened arrays

Intended to be used for either inputs or outputs
"""
class SampleData():
    

    def __init__(self, categories, sizes, scales=None, precomp=None, selected_filedata = {}):

        #NOTE: precomp dimensions should override sizes, or at least assert
        assert len(categories) == len(list(sizes.keys()))

        self.data = {}
        self.inds = {}
        self.inds_inv = {}
        self.scales = {}
        self.selected_filedata = selected_filedata

        # initialize and generate full array indices
        c  = 0
        # for cat, size in zip(categories, sizes):
        for cat in categories:
            self.data[cat] = np.empty([sizes[cat], 0])
            self.inds[cat] = list(range(c, c+sizes[cat]))
            self.selected_filedata[cat] = []
            
            #NOTE: inds dictionary entries should be unique, so inverting should work
            c2 = 0
            for ind in self.inds[cat]:
                self.inds_inv[ind] = [cat, c2]
                c2+=1

            c = c+sizes[cat]


        # if we have precomputed data in a similar dict, try adding it now
        if isinstance(precomp, dict):
            self.addData(precomp)

        # if precomp is a numpy array
        if isinstance(precomp, np.ndarray):
            self.addDataArray(precomp)

        if isinstance(scales, dict):
            assert categories == list(scales.keys())
            self.scales = scales
            

        # possible references to either inputs that generated this data, or outputs generated from this data
        self.input_ref = None
        self.output_ref = None


#------------------------------------------------------------------------------

    def __call__(self, i, cat=None):

        #return sample i, which may be a list
        
        # return full dict if category unspecified
        if cat is None:    
            sample = {}

            for key in self.data.keys():
                sample[key] = self.data[key][:, i]

            return sample
        
        # otherwise return array
        else:
            return self.data[cat][:,i]
#------------------------------------------------------------------------------

    def createData(self, N, scale=None, method='sobol'):

        # scale is dict
        if scale is not None and self.scales is None:
            self.scales = scale
            # add_scale = uniformToDist(add, self.scales, self.inds)
        elif self.scales is not None:
            if scale is not None:
                print("Distribution mapping already present")

        # uniform hypercube
        else:
            # add_scale = add
            print("No scales present, set self.scales!")
            return 

        # distinguish between newly generated random samples, and samples pulled
        # from files

        ndimtotal = self.getNDim()

        r_gen = []
        f_gen = []

        ndim = 0
        for key in self.scales.keys():
            if self.scales[key]['dist'] != "inferred":
                r_gen.append(key)
                ndim += self.getNDim(key)
            # get from file
            else:
                f_gen.append(key)

        r_dict = {key: self.scales[key] for key in r_gen}
        r_ind = {key: self.inds[key] for key in r_gen}
        f_dict = {key: self.scales[key] for key in f_gen}
        f_ind = {key: self.inds[key] for key in f_gen}

        # generate samples internally
        # ndim = self.getNDim()

        add = np.zeros([ndimtotal, N])
        # random data
        if ndim > 0:
            if method == 'sobol':
                add_raw = sobolSampleGen(ndim, N).T
            else: # elif method == 'mc': #monte carlo

                add_raw = mcSampleGen(ndim, N)

            im, ip = 0, 0
            for key, item in r_dict.items():
                ip = im + self.getNDim(key)
                add[r_ind[key], :] = add_raw[im:ip, :]
                im = ip + 0

            add = uniformToDist(add, r_dict, r_ind)

        # samples from file
        for key, item in f_dict.items():
            fname = item["datadir"]

            with open(fname, 'rb') as f:
                arr = np.load(f)

            # Nchoice = arr.shape[0]
            Nchoice = list(range(arr.shape[0]))

            Nchoice = [x for x in Nchoice if x not in self.selected_filedata[key]]
            Nind = np.random.choice(Nchoice, N, replace=True)
            # breakpoint()
            # try:
            #     Nind = np.random.choice(Nchoice, N, replace=False)
            # except:
            #     breakpoint()
            self.selected_filedata[key] = Nind
            
            add_file = arr[Nind,:].T
            add[f_ind[key], :] = add_file[:self.getNDim(key),:]


        self.addDataArray(add)
        
#------------------------------------------------------------------------------

    def addData(self, add):
        # essentially just copying a dict over and appending it
        assert isinstance(add, dict)

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

    def genSobolData(self, SD_B_pre=None):
        """
        Given the current sample data as A, generate 
        """
        cats = self.getCategories()
        sizes = {x: self.getNDim(x) for x in cats}

        # SD_A is the current instance
        SD_A = self

        # independent sample for B
        if SD_B_pre is None:
            # SD_B = SampleData(cats, sizes, precomp=sobolSampleGen(self.getNDim(), self.getNSamples()).T)
            SD_B = SampleData(cats, sizes, scales=self.scales, selected_filedata=SD_A.selected_filedata)
            SD_B.createData(N=self.getNSamples())
        else:
            SD_B = SD_B_pre

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
    if isinstance(i, int):
        A_ind = [i % N]
        B_ind = [i // N]
    else:
        A_ind = [j % N for j in list(i)]
        B_ind = [j // N for j in list(i)]

    #return sample i, which may be a list
        
    sample = {}

    for key in SD_A.data:
        sample[key] = SD_A.data[key][:, A_ind]

    # replace B entries
    B = SD_B.getInputArray()

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
    for key, ind, in inds.items():
        scale_info = scales[key]

        if "lbound" not in scale_info.keys():
            scale_info["lbound"] = len(ind)*[None]

        if "rbound" not in scale_info.keys():
            scale_info["rbound"] = len(ind)*[None]

        if isinstance(scale_info["loc"], list):
            for i in range(len(ind)):

                data_s[ind[i], :] = transformDist(data[ind[i], :], 
                                scale_info['dist'], scale_info['loc'][i], scale_info['scale'][i], scale_info["lbound"][i], scale_info["rbound"][i])

        else: # apply same dist to all variables in category

            for i in range(len(ind)):
                data_s[ind[i], :] = transformDist(data[ind[i], :], 
                                scale_info['dist'], scale_info['loc'], scale_info['scale'], scale_info['lbound'][0], scale_info["rbound"][0])
                
    return data_s



# NOTE: no beta dist capabilities yet
def transformDist(x, dist, loc, scale, a=None, b=None):

    if dist == "lognormal":
        # assume parameters are all already transformed
        if a == 0.:
            al = None
        else:
            al = np.log(a)

        res = np.exp(transformDist(x, 'normal', loc, scale, a=al, b=b))
        return res

    # if a is not None and b is not None:
    #     x_s = dist_map[dist].ppf(x, loc=loc, scale=scale, a=a, b=b)
    if a is None:
        a_use = -np.inf
    else:
        a_use = a
    if b is None:
        b_use = np.inf
    else:
        b_use = b

    if dist == 'normal':
        if a_use is None and b_use is None:
            x_s = dist_map[dist].ppf(x, loc=loc, scale=scale)
        else: # truncated normal
            x_s = dist_map["tnormal"].ppf(x, loc=loc, scale=scale, a=(a - loc) / scale, b=(b_use - loc) / scale)
    elif dist == 'fnormal': # c is the mean of the unfolded distribution, take as a, loc is the fold
        assert a_use > -np.inf, "Unfolded mean required for folded normal distribution specification"
        if b_use < np.inf:
            b_use = b
        else:
            b_use = 1

        x_s = dist_map[dist].ppf(x, b_use*(a_use - loc)/scale, loc=loc, scale=scale)
        if b_use < 0:
            x_s = -x_s + 2.0*loc 
    else: # assume beta
        x_s = dist_map[dist].ppf(x, loc=loc, scale=scale, a=a_use, b=b_use)

    # if a is not None and b is None:
    #     if dist == 'normal':
    #         x_s = dist_map["tnormal"].ppf(x, loc=loc, scale=scale, a=(a - loc) / scale, b=np.inf)
    #     else: # assuming it's beta
    #         x_s = dist_map[dist].ppf(x, loc=loc, scale=scale, a=a, b=np.inf)
    # elif b is not None and a is None:
    #     if dist == 'normal':
    #         x_s = dist_map["tnormal"].ppf(x, loc=loc, scale=scale, a=-np.inf, b=(b - loc) / scale)
    #     else: # assuming it's beta
    #         x_s = dist_map[dist].ppf(x, loc=loc, scale=scale, a=-np.inf, b=b)
    # elif a is not None and b is not None:
    #     if dist == 'normal':
    #         x_s = dist_map["tnormal"].ppf(x, loc=loc, scale=scale, a=(a - loc) / scale, b=(b - loc) / scale)
    #     else: # assuming it's beta
    #         x_s = dist_map[dist].ppf(x, loc=loc, scale=scale, a=a, b=b)
    # else:
    #     x_s = dist_map[dist].ppf(x, loc=loc, scale=scale)
    return x_s
    


"""
From parallel OpenMDAO beam problem example

Divide up samples among available procs.
"""
def divide_cases(ncases, nprocs):

    data = []
    for j in range(nprocs):
        data.append([])

    wrap = 0
    for j in range(ncases):
        idx = j - wrap
        if idx >= nprocs:
            idx = 0
            wrap = j

        data[idx].append(j)

    return data

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    # fold = 2. # fold location


    # sigma = 3.
    # #sigma = 2.
    # #sigma = 1.5 # unfolded sigma

    # nmean = 8. # unfolded mean

    # Ns = 1000

    # # transform for shape parameter
    # c = (nmean - fold)/sigma# - np.sqrt(sigma)

    # # "lbound" is c``
    # distprop = {"dist":"fnormal", "model":False, "loc":fold, "scale": sigma, "lbound": [nmean]}

    # data = SampleData(['test'], {'test': 1}, {'test': distprop})
    # data.createData(Ns)

    # # samples
    # x_s = data.data['test'][0]

    # # plot trunc norm dist
    # x = np.linspace(0., 15., 10000)
    # p = foldnorm.pdf(x, c, loc=fold, scale=sigma)
    # # p = foldnorm.pdf(x, 1.5, loc=1, scale=1)

    # plt.plot(x, p, 'k')
    # plt.hist(x_s, density=True, bins=20)

    # plt.savefig("foldnorm.png", bbox_inches="tight")

    # plt.clf()

    # also test reverse direction
    fold = 10. # fold location
    # fold = 1000000. # fold location

    sigma = 1.5
    #sigma = 2.
    #sigma = 1.5 # unfolded sigma

    nmean = 7. # unfolded mean

    Ns = 1000

    # transform for shape parameter
    c = (nmean - fold)/sigma# - np.sqrt(sigma)

    # "lbound" is c``
    # "rbound" is 1 or -1, indicates direction
    distprop = {"dist":"fnormal", "model":False, "loc":fold, "scale": sigma, "lbound": [nmean], "rbound": [-1]}

    data = SampleData(['test'], {'test': 1}, {'test': distprop})
    data.createData(Ns)

    # samples
    x_s = data.data['test'][0]

    # plot trunc norm dist
    x = np.linspace(0., 15., 10000)
    # reflect x about fold to get the actual pdf
    xf = -x + 2*fold
    p = foldnorm.pdf(xf, -c, loc=fold, scale=sigma) 

    plt.plot(x, p, 'k')
    plt.hist(x_s, density=True, bins=20)

    plt.savefig("foldnormrev.png", bbox_inches="tight")
    breakpoint()