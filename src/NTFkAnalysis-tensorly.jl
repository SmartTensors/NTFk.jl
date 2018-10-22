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
"""
function tlanalysis(T::Array, crank::Vector; seed::Number=1, backend="tensorflow", functionname::String="non_negative_tucker", init::String="svd", maxiter::Integer=DMAXITER, tol::Number=1e-4, verbose::Bool=false)
	tensorly[:set_backend](backend)
	core, factors = tensorlydecomp[Symbol(functionname)](tensorly[:backend][:tensor](T), rank=crank, n_iter_max=maxiter, init=init, tol=tol, verbose=verbose);
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	TT = TensorDecompositions.Tucker((factors...,), core)
	return TT
end