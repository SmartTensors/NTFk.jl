import NTFk

Random.seed!(1)
csize = (2, 3, 4)
tsize = (7, 667, 3038)
# tsize = (5, 10, 15)
tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(tucker_orig)
T_max = maximum(T_orig)
map!(x -> x / T_max, tucker_orig.core, tucker_orig.core)
map!(x -> x / T_max, T_orig, T_orig)

# T = NTFk.add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [tsize, csize, (1,3,4), (3,3,4), (2,2,4), (2,4,4), (2,3,3), (2,3,5)]
NTFk.analysis(T, sizes; progressbar=true, tol=1e-8, max_iter=1000)