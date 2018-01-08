import dNTF

srand(1)
csize = (2, 3, 4)
tsize = (5, 10, 15)
tucker_orig = dNTF.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(tucker_orig)
T_orig .*= 1000

# T = dNTF.add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [csize, (1,3,4), (3,3,4), (2,2,4), (2,4,4), (2,3,3), (2,3,5)]
dNTF.analysis(T, sizes, 10; progressbar=true, tol=1e-6, max_iter=1000)