import pyreadr
import pandas as pd
import numpy as np

result = pyreadr.read_r('/Users/oleg.vlasovetc/Desktop/amgut1.filt.rda')

data = result.values()
data = np.array(list(data))
data = pd.DataFrame(data[0, :, :])

# d = data.shape[1]
# n = data.shape[0]
# e = d

# import rpy2
# import os
#
# os.environ['R_HOME'] = '/path/to/R'
# import rpy2.interactive as r
# import rpy2.robjects.packages as rpackages

np.cov(data).shape

X = np.random.multivariate_normal(mean=np.zeros(289), cov=np.cov(data), size=100)
X.shape
