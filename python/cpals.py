import logging
import numpy as np
from scipy.io.matlab import loadmat
from sktensor import dtensor, cp_als

# Set logging to DEBUG to see CP-ALS information
logging.basicConfig(level=logging.DEBUG)

# Load Matlab data and convert it to dense tensor format
# mat = loadmat('/Users/monty/Julia/dNTF.py/brod.mat')

X = np.zeros((2, 10, 5))
X[:,0:2,0]=1
X[:,2:4,1]=1
X[:,4:6,2]=1
X[:,6:8,3]=1
X[:,8:10,4]=1

T = dtensor(X)

# Decompose tensor using CP-ALS
P, fit, itr, exectimes = cp_als(T, 4, init='random')
