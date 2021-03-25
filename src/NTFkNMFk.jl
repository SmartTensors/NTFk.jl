import NMFk
import DocumentFunction

function matrixfactorization(X::AbstractArray{T,N}, range::Union{AbstractRange{Int},Integer}, dims::Union{AbstractRange{Int},Integer}=1:N, aw...; kw...) where {T,N}
	@assert maximum(dims) <= N
	NTFk.hosvd(X)
	M = Vector{Tuple}(undef, N)
	for d = dims
		M[d] = NMFk.execute(NTFk.flatten(X, d), range, aw...; kw...)
	end
	return M
end

function matrixfactorization(X::AbstractArray{T,N}, range::AbstractVector, aw...; kw...) where {T,N}
	@assert length(range) == N
	NTFk.hosvd(X)
	M = Vector{Tuple}(undef, N)
	for d = 1:N
		if length(range[d]) > 0
			M[d] = NMFk.execute(NTFk.flatten(X, d), range[d], aw...; kw...)
		end
	end
	return M
end

@doc """
Matrix Factorization

$(DocumentFunction.documentfunction(matrixfactorization))
""" matrixfactorization