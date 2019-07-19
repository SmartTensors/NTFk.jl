import TensorDecompositions
import NTFk
import Random

Random.seed!(1)
trank = 3
for m = 1:5
	tsize = (5 * m, 10 * m, 20 * m)
	factors_orig = NTFk.rand_candecomp(trank, tsize; lambdas_nonneg=true, factors_nonneg=true)
	T = NTFk.compose(factors_orig)
	tranks = [1, 2, 3, 4, 5]
	for t in tranks
		factors_initial_guess = tuple([randn(dim, t) for dim in tsize]...)
		@info("CP Decomposition: Tensor rank $t")
		@time factors = TensorDecompositions.candecomp(T, t, factors_initial_guess; compute_error=true, method=:ALS)
		@time T_est = NTFk.compose(factors)
	end
end