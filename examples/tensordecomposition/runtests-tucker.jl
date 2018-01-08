import dNTF

srand(1)
csize = (2, 3, 4)
tsize = (5, 10, 15)
tucker_orig = dNTF.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(tucker_orig)
T_max = maximum(T_orig)
map!(x -> x / T_max, tucker_orig.core, tucker_orig.core)
map!(x -> x / T_max, T_orig, T_orig)

# T = dNTF.add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [csize, (1,3,4), (3,3,4), (2,2,4), (2,4,4), (2,3,3), (2,3,5)]
dNTF.analysis(T, sizes; progressbar=true)