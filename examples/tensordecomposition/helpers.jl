import TensorDecompositions

function diagonal_tucker{N}(core_dims::NTuple{N, Int}, dims::NTuple{N, Int}; core_nonneg::Bool=false, factors_nonneg::Bool=false)
	cdim = maximum(core_dims)
	@assert unique(sort(collect(core_dims)))[1] == cdim
	ndim = length(core_dims)
	diagonal_core = zeros(core_dims)
	for i=1:cdim
		ii = convert(Vector{Int64}, ones(ndim) .* i)
		diagonal_core[ii...] = 1
	end
	rnd_factor = factors_nonneg ? x -> abs.(randn(x...)) : randn
	return TensorDecompositions.Tucker((Matrix{Float64}[rnd_factor((dims[i], core_dims[i])) for i in 1:N]...), diagonal_core)
end

function rand_tucker{N}(core_dims::NTuple{N, Int}, dims::NTuple{N, Int}; core_nonneg::Bool=false, factors_nonneg::Bool=false)
	rnd_factor = factors_nonneg ? x -> abs.(randn(x...)) : randn
	rnd_core = core_nonneg ? x -> abs.(randn(x...)) : randn
	return TensorDecompositions.Tucker((Matrix{Float64}[rnd_factor((dims[i], core_dims[i])) for i in 1:N]...), rnd_core(core_dims))
end

function rand_candecomp{N}(r::Int64, dims::NTuple{N, Int}; lambdas_nonneg::Bool=false, factors_nonneg::Bool=false)
	rnd_factor = factors_nonneg ? x -> abs.(randn(x...)) : randn
	rnd_lambda = lambdas_nonneg ? x -> abs.(randn(x...)) : randn
	return TensorDecompositions.CANDECOMP((Matrix{Float64}[rnd_factor((s, r)) for s in dims]...), rnd_lambda(r))
end

rand_kruskal3{N}(r::Int64, dims::NTuple{N, Int}, nonnegative::Bool) =
	compose(rand_candecomp(r, dims, lambdas_nonneg=nonnegative, factors_nonneg=nonnegative))

function add_noise{T, N}(tnsr::Array{T,N}, sn_ratio = 0.6, nonnegative::Bool = false)
	tnsr_noise = randn(size(tnsr)...)
	if nonnegative
		map!(x -> max(0.0, x), tnsr_noise, tnsr_noise)
	end
	tnsr + 10^(-sn_ratio/0.2) * vecnorm(tnsr) / vecnorm(tnsr) * tnsr_noise
end
