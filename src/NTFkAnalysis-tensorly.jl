import PyCall

const tensorly = PyCall.PyNULL()
const tensorlydecomp = PyCall.PyNULL()
function __init__()
	try
		copy!(tensorly, PyCall.pyimport("tensorly"))
		@info("TensorLy is available")
	catch
		@warn("TensorLy is not available")
	end
	try
		copy!(tensorlydecomp, PyCall.pyimport("tensorly.decomposition"))
		@info("TensorLy.decomposition is available")
	catch
		@warn("TensorLy.decomposition is not available")
	end
end

"""
functionname = "non_negative_tucker", "non_negative_cp"
backend = "tensorflow", "pytorch", "mxnet", "numpy"
converter = :numpy, :numpy, :asnumpy, (conversion not needed)
"""
function tlanalysis(X::Array{T,N}, crank::NTuple{N, Int}; seed::Number=1, backend="tensorflow", converter=:numpy, functionname::AbstractString="non_negative_tucker", init::String="svd", maxiter::Integer=DMAXITER, tol::Number=1e-4, verbose::Bool=false) where {T, N}
	tensorly[:set_backend](backend)
	@info("Tensorly backend: $backend")
	core, factors = tensorlydecomp[Symbol(functionname)](tensorly[:backend][:tensor](X), rank=[crank...], n_iter_max=maxiter, init=init, tol=tol, verbose=verbose);
	nc = length(factors)
	f = Vector{Array}(undef, nc)
	if backend != "numpy"
		converter = backend == "mxnet" ? :asnumpy : converter
		core =  convert(Array{T,N}, core[converter]())
		for i = 1:nc
			f[i] = convert(Matrix{T}, factors[i][converter]())
		end
	else
		core =  convert(Array{T,N}, core)
		for i = 1:nc
			f[i] = convert(Matrix{T}, factors[i])
		end
	end
	# @show maximum(X .- Xe)
	TT = TensorDecompositions.Tucker((f...,), core)
	# Xe = TensorDecompositions.compose(TT)
	# @show maximum(X .- Xe)
	return TT
end