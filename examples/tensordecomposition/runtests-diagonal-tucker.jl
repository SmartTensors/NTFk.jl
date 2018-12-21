import NTFk

Random.seed!(1)
csize = (3, 3, 3)
tsize = (5, 10, 15)
# tsize = (5, 10, 15)
tucker_orig = NTFk.diagonal_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(tucker_orig)
T_max = maximum(T_orig)
map!(x -> x / T_max, tucker_orig.core, tucker_orig.core)
map!(x -> x / T_max, T_orig, T_orig)

# T = NTFk.add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [tsize, csize, (1,1,1), (2,2,2), (4,4,4)]
NTFk.analysis(T, sizes; progressbar=true, tol=1e-8, max_iter=1000)
NTFk.analysis(T, [1,2,3,4], 10; tol=1e-8, max_iter=1000)
NTFk.analysis(T, [1,2,3,4], 10; tol=1e-8, max_iter=1000, method=:cp_nmu)