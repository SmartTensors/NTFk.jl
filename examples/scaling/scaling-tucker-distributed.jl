import Distributed
Distributed.addprocs(4)

import NTFk
import TensorDecompositions
import DistributedArrays
import Random

Random.seed!(1)
trank = (2, 3, 4)
for m = 1:5
	tsize = (2 * m, 3 * m, 4 * m)
	tucker_orig = NTFk.rand_tucker(trank, tsize; core_nonneg=true, factors_nonneg=true)
	T = NTFk.compose(tucker_orig)
	dT = TensorDecompositions.distribute(T)
	tranks = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 3, 4), (3, 4, 5)]
	for t in tranks
		@info("TuckerSPNN Distributed Decomposition: Tensor core-size $t")
		@time tucker = TensorDecompositions.spnntucker(dT, t)
		@time T_est = NTFk.compose(tucker)
	end
end

A = [[1.3014 1.29658 0.729915; 0.0825359 0.0817141 0.0484702]
       [3.44648 3.42641 1.85548; 0.244351 0.244102 0.131122]
       [2.55719 2.55084 1.3693; 0.173307 0.175002 0.0914875]
       [3.60128 3.57325 1.94395; 0.253709 0.253702 0.135801]]
T = reshape(A, (2,3,4))
dest = Array{Float64,3}(undef, (1,3,4))
mtx = permutedims([0.413459 -1.48505])
TensorOperations.contract!(1, T, :N, mtx, :N, 0, dest, (2, 3), (1,), (2,), (1,), (3, 1, 2))