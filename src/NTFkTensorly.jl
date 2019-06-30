import PyCall

const tensorly = PyCall.PyNULL()
function __init__()
	try
		copy!(tensorly, PyCall.pyimport("tensorly"))
		PyCall.pyimport("tensorly.decomposition")
		@info("TensorLy is available")
	catch
		@warn("TensorLy is not available")
	end
end

"""
TensorLy Nonnegative Tucker/CP deconstruction

functionname = "non_negative_tucker", "non_negative_cp"
backend = "tensorflow", "pytorch", "mxnet", "numpy"
converter = "numpy", "numpy", "asnumpy", (conversion not needed)
"""
function tlanalysis(X::Array{T,N}, crank::NTuple{N, Int}; seed::Number=1, backend="tensorflow", converter="numpy", functionname::AbstractString="non_negative_tucker", init::String="svd", maxiter::Integer=DMAXITER, tol::Number=1e-4, verbose::Bool=false) where {T, N}
	if NTFk.tensorly == PyCall.PyNULL()
		@warn("TensorLy is not available")
		return nothing
	end
	try
		tensorly.set_backend(backend)
		@info("Tensorly backend: $backend")
	catch
		@warn("Tensorly backend $backend is not available!")
		return nothing
	end
	func = eval(:(tensorly.decomposition.$functionname))
	core, factors = func(tensorly.backend.tensor(X), rank=[crank...], n_iter_max=maxiter, init=init, tol=tol, verbose=verbose);
	nc = length(factors)
	f = Vector{Array}(undef, nc)
	if backend != "numpy"
		converter = backend == "mxnet" ? "asnumpy" : converter
		converterfield = eval(:($core.$converter))
		core =  convert(Array{T,N}, converterfield())
		for i = 1:nc
			converterfield = eval(:($factors[$i].$converter))
			f[i] = convert(Matrix{T}, converterfield())
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