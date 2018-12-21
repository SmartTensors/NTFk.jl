import NTFk

Random.seed!(2)
trank = 3
tsize = (10, 20, 5)
cp_orig = NTFk.rand_candecomp(trank, tsize; lambdas_nonneg=true, factors_nonneg=true)
T_orig = TensorDecompositions.compose(cp_orig)

# T = add_noise(T_orig, 0.6, true)
T = T_orig

tranks = [1, 2, 3, 4]
cp, _ = NTFk.analysis(T, tranks, 3, method=:ALS)
cp, _ = NTFk.analysis(T, tranks, 3, method=:cp_nmu)
cp, _ = NTFk.analysis(T, tranks, 3, method=:bcu_ncp)