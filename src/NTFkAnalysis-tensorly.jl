import PyCall

const tensorly = PyCall.PyNULL()
const tensorlydecomp = PyCall.PyNULL()
function __init__()
	try
		copy!(tensorly, PyCall.pyimport("tensorly"))
		info("TensorLy is available")
	catch
		warn("TensorLy is not available")
	end
	try
		copy!(tensorlydecomp, PyCall.pyimport("tensorly.decomposition"))
		info("TensorLy.decomposition is available")
	catch
		warn("TensorLy.decomposition is not available")
	end
end

"""
functionname = "non_negative_tucker", "non_negative_cp"
backend = "tensorflow", "pytorch", "mxnet", "numpy"
converter = :numpy, :numpy, :asnumpy
"""
function tlanalysis(X::Array{T,N}, crank::Vector; seed::Number=1, backend="tensorflow", converter=:numpy, functionname::String="non_negative_tucker", init::String="svd", maxiter::Integer=DMAXITER, tol::Number=1e-4, verbose::Bool=false) where {T, N}
	tensorly[:set_backend](backend)
	core, factors = tensorlydecomp[Symbol(functionname)](tensorly[:backend][:tensor](X), rank=crank, n_iter_max=maxiter, init=init, tol=tol, verbose=verbose);
	nc = length(factors)
	f = Vector{Array}(nc)
	if backend != "numpy"
		converter = backend == "mxnet" ? :asnumpy : converter
		core = core[converter]()
		for i = 1:nc
			f[i] = factors[i][converter]()
		end
	end
	TT = TensorDecompositions.Tucker((f...,), core)
	# Xe = TensorDecompositions.compose(TT)
	# @show maximum(X .- Xe)
	return TT
end