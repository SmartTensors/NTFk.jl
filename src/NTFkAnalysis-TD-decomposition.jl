import TensorDecompositions

function candecomp(X::AbstractArray{T, N}, r::Integer; tsize=size(X), seed::Number=1, method::Symbol=:ALS, functionname::String=string(method), tol::Float64=1e-8, maxiter::Integer=DMAXITER, compute_error::Bool=true, verbose::Bool=false, kw...) where {T,N}
	if contains(functionname, "cp_")
		c = ttanalysis(X, r; seed=abs(rand(Int16)), functionname=functionname, maxiter=maxiter, tol=tol, kw...)
	elseif contains(functionname, "bcu_")
		c = bcuanalysis(X, r; seed=abs(rand(Int16)), functionname=split(functionname, "bcu_")[2], maxiter=maxiter, tol=tol, kw...)
	else
		factors_initial_guess = tuple([randn(T, d, r) for d in tsize]...)
		c = TensorDecompositions.candecomp(X, r, factors_initial_guess; verbose=verbose, compute_error=compute_error, maxiter=maxiter, method=method)
	end
	return c
end