import logging
import numpy as np
import matplotlib.pyplot as plt
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
P, fit, itr, exectimes = cp_als(T, 5)

T_est = np.einsum('az,bz,cz->abc', P.U[0], P.U[1], P.U[2])

plt.imshow(T_est[:,:,0]);
plt.colorbar()
plt.show()
