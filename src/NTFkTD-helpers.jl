import TensorDecompositions
import Statistics

randomarray(dims, nonneg::Bool=true) = nonneg ? abs.(rand(dims...)) : rand(dims...)

function diagonal_tucker(core_dims::NTuple{N, Int}, dims::NTuple{N, Int}; core_nonneg::Bool=true, factors_nonneg::Bool=true) where {N}
	cdim = maximum(core_dims)
	@assert unique(sort(collect(core_dims)))[1] == cdim
	ndim = length(core_dims)
	diagonal_core = zeros(core_dims)
	for i=1:cdim
		ii = convert(Vector{Int64}, ones(ndim) .* i)
		diagonal_core[ii...] = 1
	end
	f = ntuple(i->randomarray((dims[i], core_dims[i]), factors_nonneg), N)
	return TensorDecompositions.Tucker(f, diagonal_core)
end

function rand_tucker(core_dims::NTuple{N, Int}, dims::NTuple{N, Int}; core_nonneg::Bool=true, factors_nonneg::Bool=true) where {N}
	f = ntuple(i->randomarray((dims[i], core_dims[i]), factors_nonneg), N)
	c = randomarray(core_dims, core_nonneg)
	return TensorDecompositions.Tucker(f, c)
end

function rand_candecomp(r::Int64, dims::NTuple{N, Int}; lambdas_nonneg::Bool=true, factors_nonneg::Bool=true) where {N}
	f = ntuple(i->randomarray((dims[i], r), factors_nonneg), N)
	return TensorDecompositions.CANDECOMP(f, randomarray(r, lambdas_nonneg))
end

rand_kruskal3(r::Int64, dims::NTuple{N, Int}, nonnegative::Bool=true) where {N} =
	TensorDecompositions.compose(rand_candecomp(r, dims, lambdas_nonneg=nonnegative, factors_nonneg=nonnegative))

function add_noise(tnsr::AbstractArray{T,N}, sn_ratio = 0.6, nonnegative::Bool = true) where {T, N}
	tnsr_noise = randn(size(tnsr)...)
	if nonnegative
		map!(x -> max(0.0, x), tnsr_noise, tnsr_noise)
	end
	tnsr + 10^(-sn_ratio/0.2) * LinearAlgebra.norm(tnsr) / LinearAlgebra.norm(tnsr) * tnsr_noise
end

function arrayoperation(A::AbstractArray{T,N}, tmap=ntuple(k->(Colon()), N), functionname="Statistics.mean") where {T, N}
	@assert length(tmap) == N
	nci = 0
	for i = 1:N
		if tmap[i] != Colon()
			if nci == 0
				nci = i
			else
				@warn("Map ($(tmap)) is wrong! More than one non-colon fields! Operation failed!")
				return
			end
		end
	end
	if nci == 0
		@warn("Map ($(tmap)) is wrong! Only one non-colon field is needed! Operation failed!")
		return
	end
	el = tmap[nci]
	asize = size(A)
	v = vec(collect(1:asize[nci]))
	deleteat!(v, el[2:end])
	t = ntuple(k->(k == nci ? v : Colon()), N)
	B = A[t...]
	t = ntuple(k->(k == nci ? el[1] : Colon()), N)
	B[t...] = Core.eval(NTFk, Meta.parse(functionname))(A[tmap...], nci)
	return B
end

function unfold(X::AbstractArray{T, N}, dim::Int=1) where {T, N}
	@assert dim > 0 && dim <= N
	i = [i for i = 1:N]
	k = i .!= dim
	TensorDecompositions._unfold(X, [dim], i[k])
end

function unfold(X::DistributedArrays.DArray{T,N,Array{T,N}}, dim::Int=1) where {T, N}
	@assert dim > 0 && dim <= N
	i = [i for i = 1:N]
	k = i .!= dim
	TensorDecompositions._unfold(X, [dim], i[k])
end

function fold(X::AbstractArray{T, N}, dim::Int, tsize::Union{NTuple{NN, Int},AbstractVector{Int}}) where {T, N, NN}
	l = length(tsize)
	@assert dim > 0 && dim <= l
	@assert N == l - 1
	t = ntuple(i->(i==dim) ? size(X, 1) : tsize[i], l)
	reshape(X, t)
end