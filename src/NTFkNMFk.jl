import NMFk
import DocumentFunction

function matrixfactorization(X::AbstractArray{T,N}, range::Union{AbstractRange{Int},Integer}, dims::Union{AbstractRange{Int},Integer}=1:N, aw...; casefilename="tensor-nmfk", kw...) where {T <: Number, N}
	@assert maximum(dims) <= N
	M = Vector{Tuple}(undef, N)
	for d = dims
		m = NMFk.flatten(X, d)
		@info("Dimension $d in range $(range[d]) with size $(size(m)):")
		M[d] = NMFk.execute(m, range[d], aw...; casefilename=casefilename * "_dim_$(d)", kw...)
end
	return M
end

function matrixfactorization(X::AbstractArray{T,N}, range::AbstractVector, aw...; casefilename="tensor-nmfk", kw...) where {T <: Number, N}
	@assert length(range) == N
	M = Vector{Tuple}(undef, N)
	for d = 1:N
		if length(range[d]) > 0
			m = NMFk.flatten(X, d)
			@info("Dimension $d in range $(range[d]) with size $(size(m)):")
			M[d] = NMFk.execute(m, range[d], aw...; casefilename=casefilename * "_dim_$(d)", kw...)
		end
	end
	return M
end

@doc """
Matrix Factorization

$(DocumentFunction.documentfunction(matrixfactorization))
""" matrixfactorization