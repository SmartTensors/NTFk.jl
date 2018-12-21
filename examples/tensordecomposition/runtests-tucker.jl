import NTFk

Random.seed!(1)
csize = (2, 3, 4)
tsize = (5, 10, 15)
tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(tucker_orig)
T_orig .*= 1000

# T = NTFk.add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [csize, (1,3,4), (3,3,4), (2,2,4), (2,4,4), (2,3,3), (2,3,5)]
tucker_spnn, ecsize, ibest = NTFk.analysis(T, sizes, 1; eigmethod=[false,false,false], progressbar=false, tol=1e-12, max_iter=10000, lambda=0.);

Random.seed!(1)
csize = (100, 10, 10)
tsize = (1000, 100, 100)
tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T = TensorDecompositions.compose(tucker_orig)

sizes = [csize]
NTFk.analysis(T, sizes, 10; progressbar=false, tol=1e-12, max_iter=10)