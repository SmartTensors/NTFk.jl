import CanDecomp
import StaticArrays
import TensorDecompositions

function janalysis(T::Array, crank::Number; seed::Number=1)
	sizes = size(T)
	ndimension = length(sizes)
	factors = Vector{Matrix{Float64}}(ndimension)
	for i = 1:ndimension
		factors[i] = zeros(crank, sizes[i]) #TODO CanDecomp.candecomp work with transposed factor matrices
	end
	CanDecomp.candecomp!(StaticArrays.SVector(factors...), deepcopy(T); regularization=0., max_cd_iters=1, tol=0.1, max_iter=3)
	TT = TensorDecompositions.CANDECOMP((factors'...), ones(crank)) # transpose factor matrices
	return TT
end

function janalysis(T::Array, crank::Vector; seed::Number=1)
	# TT = TensorDecompositions.Tucker((R["u"][1:end]...), R["core"]["data"])
	# return TT
end