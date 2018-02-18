import TensorDecompositions
import NTFk
import DistributedArrays

srand(1)
trank = 3
for m = 1:5
	tsize = (10 * m, 20 * m, 5 * m)
	factors_orig = NTFk.rand_candecomp(trank, tsize, lambdas_nonneg=true, factors_nonneg=true)
	# T = Array{Float64}(tsize)
	# Td = DistributedArrays.distribute(T)
	# T = NTFk.composeditributed!(factors_orig) # this fails
	T = TensorDecompositions.compose(factors_orig)
	Td = DistributedArrays.distribute(T)
	tranks = [1, 2, 3, 4, 5]
	for t in tranks
		factors_initial_guess = tuple([randn(dim, t) for dim in tsize]...)
		info("Tensor rank $t tensor size $tsize")
		@time factors = TensorDecompositions.candecomp(Td, t, factors_initial_guess, compute_error=true, method=:ALS)
		@time T_est = NTFk.composeditributed(factors)
	end
end