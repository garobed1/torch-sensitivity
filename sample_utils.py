from scipy.stats.qmc import scale

"""
Class for managing sample inputs between categorized dictionaries and flattened arrays

"""
class SampleData():
    

    def __init__(self, categories, sizes, precomp=None):

        #NOTE: precomp dimensions should override sizes, or at least assert

        assert len(categories) == len(sizes)

        self.data = {}
        self.inds = []

        # initialize and generate full array indices
        c  = 0
        for cat, size in categories, sizes:
            self.data[cat] = []
            self.inds.append(list(range(c, c+size)))
            c = c+size


        # self.N = N


    def getInputArray(self):
        """
        
        """

    def getSize(self):
        # get current number of samples

        return N


    def getNDim(self, key = None):
        # get current number of features

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


    def genSobolData(self):
        """
        Given the current sample data as A, 
        """

        # SD_A is the current instance
        SD_A = self

        SD_B = SampleData()




        return SD_A, SD_B, SD_AB