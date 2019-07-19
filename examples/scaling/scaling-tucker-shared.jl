import NTFk
import TensorDecompositions
import Random

Random.seed!(1)
trank = (2, 3, 4)
for m = 1:5
	tsize = (5 * m, 10 * m, 20 * m)
	tucker_orig = NTFk.rand_tucker(trank, tsize; core_nonneg=true, factors_nonneg=true)
	T = SharedArrays.SharedArray{Float64}(tsize)
	NTFk.composeshared!(T, tucker_orig)
	tranks = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 3, 4), (3, 4, 5)]
	for t in tranks
		@info("TuckerSPNN Shared Decomposition: Tensor core-size $t")
		@time tucker = TensorDecompositions.spnntucker(T, t)
		@time T_est = NTFk.compose(tucker)
	end
end