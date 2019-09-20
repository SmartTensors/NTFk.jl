import TensorDecompositions
import DocumentFunction

"""
Normalize Tucker deconstructed factors

$(DocumentFunction.documentfunction(normalizefactors!))
"""
function normalizefactors!(X::TensorDecompositions.Tucker{T,N}, order=1:N; check::Bool=false) where {T,N}
	check && (Xi = TensorDecompositions.compose(X))
	l = size(X.core)
	for i = order
		m = maximum(X.factors[i]; dims=1)
		@assert length(m) == l[i]
		for j = 1:l[i]
			ind = map(k->((i==k) ? j : Colon()), 1:N)
			X.core[ind...] .*= m[j]
		end
		m[m.==0] .= 1.0
		X.factors[i] ./= m
	end
	if check
		Xe = TensorDecompositions.compose(X)
		@info("Normalization error: $(LinearAlgebra.norm(Xi .- Xe))")
	end
	return nothing
end

"""
Normalize Tucker deconstructed core

$(DocumentFunction.documentfunction(normalizecore!))
"""
function normalizecore!(X::TensorDecompositions.Tucker{T,N}, order=1:N; check::Bool=false) where {T,N}
	check && (Xi = TensorDecompositions.compose(X))
	l = size(X.core)
	v = collect(1:N)
	for i = order
		m = vec(maximum(X.core; dims=v[v.!=i]))
		X.factors[i] .*= m'
		m[m.==0] .= 1.0
		for j = 1:l[i]
			ind = map(k->((i==k) ? j : Colon()), 1:N)
			X.core[ind...] ./= m[j]
			# @show m[j]
		end
		m = vec(maximum(X.core; dims=v[v.!=i]))
	end
	if check
		Xe = TensorDecompositions.compose(X)
		@info("Normalization error: $(LinearAlgebra.norm(Xi .- Xe))")
	end
	return nothing
end

"""
Normalize Tucker deconstructed slices

$(DocumentFunction.documentfunction(normalizeslices!))
"""
function normalizeslices!(X::TensorDecompositions.Tucker{T,N}, order=1:N; check::Bool=false) where {T,N}
	check && (Xi = TensorDecompositions.compose(X))
	NTFk.normalizefactors!(X)
	NTFk.normalizecore!(X, order)
	M = NTFk.compose(X, order[2:end])
	m = maximum(M; dims=order[2:end])
	for i = 1:length(m)
		t = ntuple(k->(k == order[1] ? i : Colon()), N)
		X.core[t...] ./= m[i]
		X.factors[order[1]][:,i] .*= m[i]
	end
	if check
		Xe = TensorDecompositions.compose(X)
		@info("Normalization error: $(LinearAlgebra.norm(Xi .- Xe))")
	end
end

"""
Scale Tucker deconstructed slices

$(DocumentFunction.documentfunction(scalefactors!))
"""
function scalefactors!(X::TensorDecompositions.Tucker{T,N}, m::AbstractVector, order=1:N; check::Bool=false) where {T,N}
	check && (Xi = TensorDecompositions.compose(X))
	NTFk.normalizefactors!(X)
	NTFk.normalizecore!(X, order)
	@assert length(m) == size(X.core, order[1])
	for i = 1:length(m)
		X.factors[order[1]][:,i] .*= m[i]
	end
	if check
		Xe = TensorDecompositions.compose(X)
		@info("Normalization error: $(LinearAlgebra.norm(Xi .- Xe))")
	end
end

"""
Normalize CP deconstructed factors

$(DocumentFunction.documentfunction(normalizefactors!))
"""
function normalizefactors!(X::TensorDecompositions.CANDECOMP{T,N}, order=1:N; check::Bool=false) where {T,N}
	check && (Xi = TensorDecompositions.compose(X))
	for i = order
		m = maximum(X.factors[i]; dims=1)
		X.lambdas .*= vec(m)
		m[m.==0] = 1.0
		X.factors[i] ./= m
	end
	if check
		Xe = TensorDecompositions.compose(X)
		@info("Normalization error: $(LinearAlgebra.norm(Xi .- Xe))")
	end
	return nothing
end

"""
Normalize CP deconstructed lambdas

$(DocumentFunction.documentfunction(normalizelambdas!))
"""
function normalizelambdas!(X::TensorDecompositions.CANDECOMP{T,N}, order=1:N; check::Bool=false) where {T,N}
	check && (Xi = TensorDecompositions.compose(X))
	m = vec(X.lambdas)' .^ (1/N)
	for i = order
		X.factors[i] .*= m
	end
	m = copy(X.lambdas)
	m[m.==0] .= 1.0
	X.lambdas ./= m
	if check
		Xe = TensorDecompositions.compose(X)
		@info("Normalization error: $(LinearAlgebra.norm(Xi .- Xe))")
	end
	return nothing
end