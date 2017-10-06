import TensorDecompositions
import dNTF

srand(1)
trank = 3
for m = 1:5
	tsize = (10 * m, 20 * m, 5 * m)
	factors_orig = dNTF.rand_candecomp(trank, tsize, lambdas_nonneg=true, factors_nonneg=true)
	T = SharedArray{Float64}(tsize)
	dNTF.composeshared!(T, factors_orig)
	tranks = [1, 2, 3, 4, 5]
	for t in tranks
		factors_initial_guess = tuple([randn(dim, t) for dim in tsize]...)
		info("Tensor rank $t tensor size $tsize")
		@time factors = TensorDecompositions.candecomp(T, t, factors_initial_guess, compute_error=true, method=:ALS)
		@time T_est = dNTF.composeshared(factors)
	end
end