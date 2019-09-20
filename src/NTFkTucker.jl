import TensorDecompositions
import Random
import Statistics
import DocumentFunction

function analysis(X::AbstractArray{T,N}, dsizes::Vector{Int64}, dim, nTF; kw...) where {T,N}
	csize = collect(size(X))
	ndimensons = length(csize)
	sizes = Vector{Tuple}(undef, 0)
	for i in dsizes
		nt = ntuple(k->(k == dim ? i : csize[k]), ndimensons)
		push!(sizes, nt)
	end
	@info("Sizes: $(sizes)")
	analysis(X, sizes, nTF; kw...)
end

"""
Tucker deconstruction: Multiple analyses for different core sizes

methods: spnntucker, tucker_als, tucker_sym, tensorly_
"""
function analysis(X::AbstractArray{T,N}, csizes::Vector{NTuple{N,Int}}, nTF::Integer=1; clusterdim::Integer=1, resultdir::String=".", prefix::String="spnn", serial::Bool=false, seed::Integer=0, kw...) where {T,N}
	@info("TensorDecompositions Tucker analysis for a series of $(length(csizes)) core sizes ...")
	@info("Clustering Dimension: $clusterdim")
	recursivemkdir(resultdir; filename=false)
	recursivemkdir(prefix; filename=false)
	@assert clusterdim <= N || clusterdim > 1
	seed > 0 && Random.seed!(seed)
	tsize = size(X)
	ndimensons = length(tsize)
	nruns = length(csizes)
	residues = Vector{T}(undef, nruns)
	correlations = Array{T}(undef, nruns, ndimensons)
	tucker_spnn = Vector{TensorDecompositions.Tucker{T,N}}(undef, nruns)
	minsilhouette = Vector{T}(undef, nruns)
	if nprocs() > 1 && !serial
		r = pmap(i->(Random.seed!(seed+i); analysis(X, csizes[i], nTF; clusterdim=clusterdim, resultdir=resultdir, prefix=prefix, kw..., serial=true, quiet=true)), 1:nruns)
		tucker_spnn = map(i->(r[i][1]), 1:nruns)
		residues = map(i->(r[i][2]), 1:nruns)
		correlations = map(i->(r[i][3]), 1:nruns)
		minsilhouette = map(i->(r[i][4]), 1:nruns)
	else
		s = nprocs() > 1 ? false : true
		for i in 1:nruns
			a = analysis(X, csizes[i], nTF; clusterdim=clusterdim, resultdir=resultdir, prefix=prefix, serial=s, kw...)
			if a != nothing
				tucker_spnn[i], residues[i], correlations[i,:], minsilhouette[i] = a
			end
		end
	end
	@info("Decompositions (clustering dimension: $clusterdim)")
	ibest = 1
	best = Inf
	for i in 1:nruns
		if residues[i] < best
			best = residues[i]
			ibest = i
		end
	end
	for i in 1:nruns
		if isassigned(tucker_spnn, i)
			println("$i - $(csizes[i]): residual $(residues[i]) worst tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core)) silhouette $(minsilhouette[i])")
		end
	end
	# NTFk.atensor(tucker_spnn[ibest].core)
	if isassigned(tucker_spnn, ibest)
		csize = TensorToolbox.mrank(tucker_spnn[ibest].core)
		@info("Estimated true core size based on the reconstruction: $(csize)")
		if prefix != ""
			if nruns > 1
				FileIO.save("$(resultdir)/$(prefix)-$(mapsize(csize)).$(outputformat)", "tucker_vector", tucker_spnn)
			else
				FileIO.save("$(resultdir)/$(prefix)-$(mapsize(csize)).$(outputformat)", "tucker", tucker_spnn[1])
			end
		end
		return tucker_spnn, csize, ibest
	else
		@warn("Execution failed!")
		return nothing
	end
end

"""
Tucker deconstruction: Single analysis of a given core size

methods: spnntucker, tucker_als, tucker_sym, tensorly_

$(DocumentFunction.documentfunction(analysis))
"""
function analysis(X::AbstractArray{T,N}, csize::NTuple{N,Int}=size(X), nTF::Integer=1; serial::Bool=false, clusterdim::Integer=1, resultdir::String=".", saveall::Bool=false, quiet::Bool=true, method=:spnntucker, prefix::String="spnn", seed::Integer=-1, kw...) where {T,N}
	if occursin("tucker_", string(method))
		@info("MATLAB TensorToolbox Tucker analysis using $(string(method)) ...")
		prefix = "tensortoolbox"
	elseif occursin("tensorly_", string(method))
		@info("Python Tensorly Tucker analysis using $(string(method)) ...")
		prefix = "tensorly"
		method = :tensorly_non_negative_tucker
	else
		@info("TensorDecompositions Sparse Nonnegative Tucker analysis using $(string(method)) ...")
	end
	@info("Core size $(csize)...")
	@info("Clustering Dimension: $clusterdim")
	@assert clusterdim <= N || clusterdim > 1
	if seed < 0
		seed = abs(rand(Int16))
		@info("Random seed: $seed")
	else
		@info("Provided seed: $seed")
	end
	Random.seed!(seed)
	tsize = size(X)
	ndimensons = length(tsize)
	@info("Tensor size: $(tsize)")
	residues = Vector{Float64}(undef, nTF)
	tsi = Vector{TensorDecompositions.Tucker{T,N}}(undef, nTF)
	WBig = Vector{Matrix}(undef, nTF)
	# nans = isnan.(X)
	# if sum(nans) > 0
	# 	@warn("The tensor has NaN's; they will be zeroed temporarily.")
	# 	X[nans] .= 0
	# end
	tsbest = nothing
	if nprocs() > 1 && !serial
		tsi = pmap(i->(Random.seed!(seed+i); NTFk.tucker(X, csize; seed=seed, method=method, kw..., progressbar=false)), 1:nTF)
	else
		for n = 1:nTF
			@time a = NTFk.tucker(X, csize; seed=seed, method=method, kw...)
			if a != nothing
				tsi[n] = a
			end
		end
	end
	for n = 1:nTF
		if isassigned(tsi, n)
			residues[n] = TensorDecompositions.rel_residue(tsi[n], X)
			println("$(n): relative residual $(residues[n])")
			normalizecore!(tsi[n])
			f = permutedims(tsi[n].factors[clusterdim])
			# f[f.==0] = max(minimum(f), 1e-6)
			# p = NTFk.plotmatrix(cpi[n].factors[1]')
			# display(p); println()
			# p = NTFk.plotmatrix(f)
			# display(p); println()
			# @show minimum(cpi[n].lambdas), maximum(cpi[n].lambdas)
			WBig[n] = hcat(f)
		else
			residues[n] = Inf
		end
	end
	# if sum(nans) > 0
	# 	X[nans] .= NaN
	# end
	minsilhouette = nTF > 1 ? clusterfactors(WBig, quiet) : NaN
	imin = argmin(residues)
	if isassigned(tsi, imin)
		X_esta = TensorDecompositions.compose(tsi[imin])
		correlations = mincorrelations(X_esta, X)
		# NTFk.atensor(tsi[imin].core)
		csize_new = TensorToolbox.mrank(tsi[imin].core)
		println("$(csize): relative residual $(residues[imin]) worst tensor correlations $(correlations) rank $(csize_new) silhouette $(minsilhouette)")
		if saveall
			recursivemkdir(resultdir; filename=false)
			recursivemkdir(prefix; filename=false)
			FileIO.save("$(resultdir)/$(prefix)-$(mapsize(csize))->$(mapsize(csize_new)).$(outputformat)", "tucker", tsi[imin])
		end
		return tsi[imin], residues[imin], correlations, minsilhouette
	else
		return nothing
	end
end

"""
Tucker deconstruction: Single analysis of a given core size

methods: spnntucker, tucker_als, tucker_sym, tensorly_

$(DocumentFunction.documentfunction(tucker))
"""
function tucker(X::AbstractArray{T, N}, csize::NTuple{N, Int}; seed::Number=0, method::Symbol=:spnntucker, functionname::String=string(method), maxiter::Integer=DMAXITER, core_nonneg::Bool=true, verbose::Bool=false, tol::Number=1e-8, ini_decomp::Symbol=:ntfk_hosvd, lambda::Number=0.1, lambdas=fill(lambda, length(size(X)) + 1), eigmethod=trues(N), eigreduce=eigmethod, progressbar::Bool=false, order=1:N, compute_error::Bool=true, compute_rank::Bool=true, whichm::Symbol=:LM, hosvd_tol::Number=0.0, hosvd_maxiter::Integer=300, rtol::Number=0., kw...) where {T,N}
	if occursin("tucker_", string(method))
		c = ttanalysis(X, csize; seed=seed, functionname=functionname, maxiter=maxiter, tol=tol, kw...)
	elseif occursin("tensorly_", string(method))
		c = tlanalysis(X, csize; seed=seed, functionname=split(functionname, "tensorly_")[2], maxiter=maxiter, tol=tol, kw...)
	else
		if ini_decomp == :ntfk_hosvd
		    nans = isnan.(X)
		    if sum(nans) > 0
		        nanflag = true
		        X[nans] .= Statistics.mean(X[.!nans])
		    else
		        nanflag = false
		    end
			ini_decomp = NTFk.hosvd(X, csize, eigmethod, eigreduce; pad_zeros=true, order=order, compute_error=compute_error, compute_rank=compute_rank, whichm=whichm, tol=hosvd_tol, maxiter=hosvd_maxiter, rtol=rtol)
		    if nanflag
		        X[nans] .= NaN
		    end
		end
		c = TensorDecompositions.spnntucker(X, csize; ini_decomp=ini_decomp, core_nonneg=core_nonneg, verbose=verbose, max_iter=maxiter, tol=tol, lambdas=lambdas)
	end
	return c
end