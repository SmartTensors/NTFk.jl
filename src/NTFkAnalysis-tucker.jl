import TensorDecompositions

"""
methods: spnntucker, tucker_als, tucker_sym
"""
function analysis(X::AbstractArray{T,N}, csize::NTuple{N,Int}=size(X), nTF::Integer=1; clusterdim::Integer=1, resultdir::String=".", prefix::String="spnn", seed::Integer=0, tol::Number=1e-8, ini_decomp=:hosvd, core_nonneg=true, verbose=false, max_iter::Integer=DMAXITER, lambda::Number=0.1, lambdas=fill(lambda, length(size(X)) + 1), eigmethod=trues(N), progressbar::Bool=false, quiet::Bool=true, serial::Bool=false, saveall::Bool=false) where {T,N}
	info("TensorDecompositions Tucker analysis with core size $(csize)...")
	info("Clustering Dimension: $clusterdim")
	@assert clusterdim <= N || clusterdim > 1
	seed > 0 && srand(seed)
	tsize = size(X)
	ndimensons = length(tsize)
	info("Tensor size: $(tsize)")
	residues = Vector{Float64}(nTF)
	tsi = Vector{TensorDecompositions.Tucker{T,N}}(nTF)
	WBig = Vector{Matrix}(nTF)
	nans = isnan.(X)
	if sum(nans) > 0
		warn("The tensor has NaN's; they will be zeroed temporarily.")
		X[nans] .= 0
	end
	tsbest = nothing
	# lambdas = convert(Vector{T}, lambdas)
	if nprocs() > 1 && !serial
		tsi = pmap(i->(srand(seed+i); TensorDecompositions.spnntucker(X, csize; eigmethod=eigmethod, tol=tol, ini_decomp=ini_decomp, core_nonneg=core_nonneg, verbose=verbose, max_iter=max_iter, lambdas=lambdas, progressbar=false)), 1:nTF)
	else
		for n = 1:nTF
			@time tsi[n] = TensorDecompositions.spnntucker(X, csize; eigmethod=eigmethod, tol=tol, ini_decomp=ini_decomp, core_nonneg=core_nonneg, verbose=verbose, max_iter=max_iter, lambdas=lambdas, progressbar=progressbar)
		end
	end
	for n = 1:nTF
		residues[n] = TensorDecompositions.rel_residue(tsi[n], X)
		println("$(n): relative residual $(residues[n])")
		normalizecore!(tsi[n])
		f = tsi[n].factors[clusterdim]'
		# f[f.==0] = max(minimum(f), 1e-6)
		# p = NTFk.plotmatrix(cpi[n].factors[1]')
		# display(p); println()
		# p = NTFk.plotmatrix(f)
		# display(p); println()
		# @show minimum(cpi[n].lambdas), maximum(cpi[n].lambdas)
		WBig[n] = hcat(f)
	end
	minsilhouette = nTF > 1 ? clusterfactors(WBig, quiet) : NaN
	imin = indmin(residues)
	X_esta = TensorDecompositions.compose(tsi[imin])
	correlations = mincorrelations(X_esta, X)
	# NTFk.atensor(tsi[imin].core)
	csize_new = TensorToolbox.mrank(tsi[imin].core)
	println("$(csize): relative residual $(residues[imin]) worst tensor correlations $(correlations) rank $(csize_new) silhouette $(minsilhouette)")
	if saveall
		recursivemkdir(resultdir; filename=false)
		recursivemkdir(prefix; filename=false)
		JLD.save("$(resultdir)/$(prefix)-$(mapsize(csize))->$(mapsize(csize_new)).jld", "t", tsi[imin])
	end

	if sum(nans) > 0
		X[nans] .= NaN
	end

	return tsi[imin], residues[imin], correlations, minsilhouette
end

function analysis(X::AbstractArray{T,N}, csizes::Vector{NTuple{N,Int}}, nTF::Integer=1; clusterdim::Integer=1, resultdir::String=".", prefix::String="spnn", serial::Bool=false, seed::Integer=0, kw...) where {T,N}
	info("TensorDecompositions Tucker analysis for a series of $(length(csizes)) core sizes ...")
	warn("Clustering Dimension: $clusterdim")
	recursivemkdir(resultdir; filename=false)
	recursivemkdir(prefix; filename=false)
	@assert clusterdim <= N || clusterdim > 1
	seed > 0 && srand(seed)
	tsize = size(X)
	ndimensons = length(tsize)
	nruns = length(csizes)
	residues = Vector{T}(nruns)
	correlations = Array{T}(nruns, ndimensons)
	tucker_spnn = Vector{TensorDecompositions.Tucker{T,N}}(nruns)
	minsilhouette = Vector{T}(nruns)
	if nprocs() > 1 && !serial
		r = pmap(i->(srand(seed+i); analysis(X, csizes[i], nTF; clusterdim=clusterdim, resultdir=resultdir, prefix=prefix, kw..., serial=true, quiet=true)), 1:nruns)
		tucker_spnn = map(i->(r[i][1]), 1:nruns)
		residues = map(i->(r[i][2]), 1:nruns)
		correlations = map(i->(r[i][3]), 1:nruns)
		minsilhouette = map(i->(r[i][4]), 1:nruns)
	else
		s = nprocs() > 1 ? false : true
		for i in 1:nruns
			tucker_spnn[i], residues[i], correlations[i,:], minsilhouette[i] = analysis(X, csizes[i], nTF; clusterdim=clusterdim, resultdir=resultdir, prefix=prefix, serial=s, kw...)
		end
	end
	info("Decompositions (clustering dimension: $clusterdim)")
	ibest = 1
	best = Inf
	for i in 1:nruns
		if residues[i] < best
			best = residues[i]
			ibest = i
		end
	end
	for i in 1:nruns
		println("$i - $(csizes[i]): residual $(residues[i]) worst tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core)) silhouette $(minsilhouette[i])")
	end
	# NTFk.atensor(tucker_spnn[ibest].core)
	csize = TensorToolbox.mrank(tucker_spnn[ibest].core)
	info("Estimated true core size based on the reconstruction: $(csize)")
	JLD.save("$(resultdir)/$(prefix)-$(mapsize(csize)).jld", "t", tucker_spnn)
	return tucker_spnn, csize, ibest
end