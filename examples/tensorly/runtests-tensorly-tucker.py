import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize
from tensorly.decomposition import non_negative_parafac
from tensorly.decomposition import non_negative_tucker
from tensorly.decomposition import tucker
from math import ceil

from tensorly.base import tensor_to_vec, partial_tensor_to_vec
from tensorly.datasets.synthetic import gen_image
from tensorly.random import check_random_state
from tensorly.regression.kruskal_regression import KruskalRegressor
import tensorly.backend as T
import time

tl.set_backend('numpy')
rng = check_random_state(1)
X = T.tensor(rng.normal(size=(1000, 1000, 1000), loc=0, scale=1))
start_time = time.time()
core, tucker_factors = non_negative_tucker(X, rank=[10,10,10], init='svd', tol=10e-12, verbose=True, n_iter_max=2)
print("--- %s seconds ---" % (time.time() - start_time))

tl.set_backend('mxnet')
rng = check_random_state(1)
X = T.tensor(rng.normal(size=(1000, 1000, 1000), loc=0, scale=1))
start_time = time.time()
core, tucker_factors = non_negative_tucker(X, rank=[10,10,10], init='svd', tol=10e-12, verbose=True, n_iter_max=2)
print("--- %s seconds ---" % (time.time() - start_time))

tl.set_backend('pytorch')
rng = check_random_state(1)
X = T.tensor(rng.normal(size=(1000, 1000, 1000), loc=0, scale=1))
start_time = time.time()
core, tucker_factors = non_negative_tucker(X, rank=[10,10,10], init='svd', tol=10e-12, verbose=True, n_iter_max=2)
print("--- %s seconds ---" % (time.time() - start_time))

tl.set_backend('tensorflow')
rng = check_random_state(1)
X = T.tensor(rng.normal(size=(1000, 1000, 1000), loc=0, scale=1))
start_time = time.time()
core, tucker_factors = non_negative_tucker(X, rank=[10,10,10], init='svd', tol=10e-12, verbose=True, n_iter_max=2)
print("--- %s seconds ---" % (time.time() - start_time))
