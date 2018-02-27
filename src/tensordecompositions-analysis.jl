import TensorDecompositions
import NMFk

DMAXITER = 1000

function loadcase(case::String; datadir::String=".")
	f = "$(datadir)/$(case)F.jld"
	if isfile(f)
		F = JLD.load(f, "X")
	else
		warn("File $f is missing")
		return nothing
	end
	f = "$(datadir)/$(case)G.jld"
	if isfile(f)
		G = JLD.load(f, "X")
	else
		warn("File $f is missing")
		return nothing
	end
	# A = max.(F .- G, 0)
	# B = max.(G .- F, 0)
	C = max.(F .- max.(F .- G, 0), 0)
	return C
end

function loadresults(case::String, csize::Tuple=(); resultdir::String=".")
	filename = "$(resultdir)/$(case)-$(csize[1])_$(csize[2])_$(csize[3]).jld"
	if isfile(filename)
		t = JLD.load(filename, "t")
		return t
	else
		warn("File $(filename) does not exist!")
		return nothing
	end
end

function analysistime1(case::String; timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, trank=10, datadir::String=".", resultdir::String=".", moviedir::String=".", figuredir::String=".", suffix::String="", seed::Number=0, max_iter=DMAXITER, tol=1e-8, ini_decomp=nothing, lambda::Number=0.1)
	if !isdir(resultdir)
		mkdir(resultdir)
	end
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	if !isdir(figuredir)
		mkdir(figuredir)
	end
	C = loadcase(case; datadir=datadir)
	if C == nothing
		return (0,0,0)
	end
	csize = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, trank=trank, datadir=datadir, resultdir=resultdir, moviedir=moviedir, figuredir=figuredir, skipmakemovies=true, lambda=0.1, problemname="sparse", seed=seed, max_iter=max_iter, tol=tol, ini_decomp=ini_decomp, lambda=lambda)
	return csize
end

function analysistime(case::String; timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, trank=10, datadir::String=".", resultdir::String=".", moviedir::String=".", figuredir::String=".", suffix::String="", seed::Number=0, max_iter=DMAXITER, tol=1e-8)
	if !isdir(resultdir)
		mkdir(resultdir)
	end
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	if !isdir(figuredir)
		mkdir(figuredir)
	end
	C = loadcase(case; datadir=datadir)
	if C == nothing
		return (0,0,0)
	end
	csize = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, trank=trank, datadir=datadir, resultdir=resultdir, moviedir=moviedir, figuredir=figuredir, lambda=0.1, problemname="sparse", seed=seed, max_iter=max_iter, tol=tol, ini_decomp=:hosvd)
	_ = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, trank=csize[1], datadir=datadir, resultdir=resultdir, moviedir=moviedir, figuredir=figuredir, lambda=0.000000001, problemname="dense", seed=seed, max_iter=max_iter, tol=tol, ini_decomp=nothing)
	return csize
end

function analysis(case::String; timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, trank1=10, trank2=3, datadir::String=".", resultdir::String=".", moviedir::String=".", suffix::String="", seed::Number=0, max_iter=DMAXITER, tol=1e-8, ini_decomp=nothing)
	if !isdir(resultdir)
		mkdir(resultdir)
	end
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	C = loadcase(case; datadir=datadir)
	if C == nothing
		return (0,0,0)
	end
	csize = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, trank=trank1, datadir=datadir, resultdir=resultdir, moviedir=moviedir, lambda=0.1, problemname="sparse", skipxymakemovies=true, seed=seed, max_iter=max_iter, tol=tol, ini_decomp=ini_decomp)
	_ = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, trank=trank2, datadir=datadir, resultdir=resultdir, moviedir=moviedir, lambda=0.000000001, problemname="dense", seed=seed, max_iter=max_iter, tol=tol, ini_decomp=ini_decomp)
	return csize
end

function analysis(case::String, X::Array, csize::Tuple=(); timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, trank=10, datadir::String=".", resultdir::String=".", moviedir::String=".", figuredir::String=".", problemname::String="sparse", makemovie::Bool=true, skipmakemovies::Bool=false, skipmakedatamovie::Bool=false, skipmaketimemovies::Bool=false, skipxymakemovies::Bool=true, quiet::Bool=true, seed::Number=0, max_iter=DMAXITER, tol=1e-8, ini_decomp=nothing, lambda::Number=0.1)
	if length(csize) == 0
		if !skipmakemovies && !skipmakedatamovie
			info("Making problem movie for $(case) ...")
			NTFk.plottensor(X[timeindex, xindex, yindex]; movie=makemovie, moviedir=moviedir, prefix="$(case)", quiet=quiet)
		end
		xrank = length(collect(xindex))
		yrank = length(collect(yindex))
		trank = trank
		info("Solving $(problemname) problem for $(case) ...")
		t, csize = NTFk.analysis(X[timeindex, xindex, yindex], [(trank, xrank, yrank)]; resultdir=resultdir, prefix="$(case)-", seed=seed, tol=tol, ini_decomp=ini_decomp, core_nonneg=true, verbose=false, max_iter=max_iter, lambda=lambda)
	else
		t = loadresults(case, csize; resultdir=resultdir)
		if t == nothing
			return csize
		end
	end
	if !skipmakemovies
		if !skipmaketimemovies
			info("Making $(problemname) problem comparison movie for $(case) ...")
			nt = TensorDecompositions.compose(t[1])
			NTFk.plotcmptensor(X[timeindex, xindex, yindex], nt; movie=makemovie, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])", quiet=quiet)
			info("Making $(problemname) problem leftover movie for $(case) ...")
			NTFk.plotlefttensor(X[timeindex, xindex, yindex], nt, X[timeindex, xindex, yindex] .- nt; movie=makemovie, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-left", quiet=quiet)
			info("Making $(problemname) problem component T movie for $(case) ...")
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 1; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-t", quiet=quiet)
		end
		info("Making $(problemname) 2D component plot for $(case) ...")
		NTFk.plot2dtensorcomponents(t[1]; quiet=quiet, filename="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-t2d.png", figuredir=figuredir)
		if !skipmaketimemovies && !skipxymakemovies
			info("Making $(problemname) problem component X movie for $(case) ...")
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 2; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-x", quiet=quiet)
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 2, 1; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-xt", quiet=quiet)
			info("Making $(problemname) problem component Y movie for $(case) ...")
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 3; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-y", quiet=quiet)
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 3, 1; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-yt", quiet=quiet)
		end
	end
	return csize
end

"""
methods: spnntucker, tucker_als, tucker_sym
"""
function analysis{T,N}(X::Array{T,N}, sizes=[size(X)], nTF=1; resultdir::String=".", prefix::String="", seed::Number=0, tol=1e-8, ini_decomp=:hosvd, core_nonneg=true, verbose=false, max_iter=DMAXITER, lambda::Number=0.1, lambdas=fill(lambda, length(size(X)) + 1), eigmethod=trues(N), progressbar::Bool=false, quiet::Bool=true)
	info("TensorDecompositions Tucker analysis ...")
	seed > 0 && srand(seed)
	tsize = size(X)
	ndimensons = length(tsize)
	nruns = length(sizes)
	residues = Array{T}(nruns)
	correlations = Array{T}(nruns, ndimensons)
	X_esta = Array{Array{T,N}}(nruns)
	tucker_spnn = Array{TensorDecompositions.Tucker{T,N}}(nruns)
	minsilhouette = Array{T}(nruns)
	for i in 1:nruns
		info("Core size: $(sizes[i])")
		residues2 = Array{Float64}(nTF)
		tsi = Array{TensorDecompositions.Tucker{T,N}}(nTF)
		WBig = Vector{Matrix}(nTF)
		tsbest = nothing
		for n = 1:nTF
			@time tsi[n] = TensorDecompositions.spnntucker(X, sizes[i]; eigmethod=eigmethod, tol=tol, ini_decomp=ini_decomp, core_nonneg=core_nonneg, verbose=verbose, max_iter=max_iter, lambdas=lambdas, progressbar=progressbar)
			residues2[n] = TensorDecompositions.rel_residue(tsi[n], X)
			normalizecore!(tsi[n])
			f = tsi[n].factors[1]'
			f[f.==0] = 1e-6
			# p = NTFk.plotmatrix(cpi[n].factors[1]')
			# display(p); println()
			# p = NTFk.plotmatrix(f)
			# display(p); println()
			# @show minimum(cpi[n].lambdas), maximum(cpi[n].lambdas)
			WBig[n] = hcat(f)
		end
		if nTF > 1
			clusterassignments, M = NMFk.clustersolutions(WBig)
			if !quiet
				info("Cluster assignments:")
				display(clusterassignments)
				info("Cluster centroids:")
				display(M)
			end
			Wa, clustersilhouettes, Wv = NMFk.finalize(WBig, clusterassignments)
			minsilhouette[i] = minimum(clustersilhouettes)
			if !quiet
				info("Silhouettes for each of the $(length(clustersilhouettes)) clusters:" )
				display(clustersilhouettes')
				println("Mean silhouette = ", mean(clustersilhouettes))
				println("Min  silhouette = ", minimum(clustersilhouettes))
			end
		else
			minsilhouette[i] = NaN
		end
		imin = indmin(residues2)
		tucker_spnn[i] = tsi[imin]
		X_esta[i] = TensorDecompositions.compose(tucker_spnn[i])
		residues[i] = TensorDecompositions.rel_residue(X_esta[i], X)
		correlations[i,:] = mincorrelations(X_esta[i], X)
		println("$i - $(sizes[i]): residual $(residues[i]) worst tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core)) silhouette $(minsilhouette[i])")
	end
	info("Decompositions:")
	ibest = 1
	best = Inf
	for i in 1:nruns
		if residues[i] < best
			best = residues[i]
			ibest = i
		end
		println("$i - $(sizes[i]): residual $(residues[i]) worst tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core)) silhouette $(minsilhouette[i])")
	end
	# NTFk.atensor(tucker_spnn[ibest].core)
	csize = TensorToolbox.mrank(tucker_spnn[ibest].core)
	info("Estimated true core size: $(csize)")
	JLD.save("$(resultdir)/$(prefix)$(csize[1])_$(csize[2])_$(csize[3]).jld", "t", tucker_spnn)
	return tucker_spnn, csize, ibest
end

"""
methods: ALS, SGSD, cp_als, cp_apr, cp_nmu, cp_opt, cp_sym, cp_wopt
"""
function analysis{T,N}(X::Array{T,N}, tranks::Vector{Int64}, nTF=1; resultdir::String=".", prefix::String="", seed::Number=-1, tol=1e-8, verbose=false, max_iter=DMAXITER, method=:ALS, quiet=true, kw...)
	if contains(string(method), "cp_")
		info("TensorToolbox CanDecomp analysis ...")
	elseif contains(string(method), "bcu_")
		info("Block-coordinate nonconvex CanDecomp analysis ...")
	else
		info("TensorDecompositions CanDecomp analysis ...")
	end
	seed >= 0 && srand(seed)
	tsize = size(X)
	ndimensons = length(tsize)
	nruns = length(tranks)
	residues = Array{T}(nruns)
	correlations = Array{T}(nruns, ndimensons)
	X_esta = Array{Array{T,N}}(nruns)
	cpf = Array{TensorDecompositions.CANDECOMP{T,N}}(nruns)
	minsilhouette = Array{Float64}(nruns)
	for i in 1:nruns
		info("CP core rank: $(tranks[i])")
		residues2 = Array{T}(nTF)
		cpi = Array{TensorDecompositions.CANDECOMP{T,N}}(nTF)
		WBig = Vector{Matrix}(nTF)
		cpbest = nothing
		for n = 1:nTF
			@time cpi[n] = NTFk.candecomp(X, tranks[i]; verbose=verbose, maxiter=max_iter, method=method, tol=tol, kw...)
			residues2[n] = TensorDecompositions.rel_residue(cpi[n], X)
			normalizelambdas!(cpi[n])
			f = map(k->abs.(cpi[n].factors[k]'), 1:ndimensons)
			# p = NTFk.plotmatrix(cpi[n].factors[1]')
			# display(p); println()
			# p = NTFk.plotmatrix(f)
			# display(p); println()
			# @show minimum(cpi[n].lambdas), maximum(cpi[n].lambdas)
			WBig[n] = hcat(f...)
		end
		if nTF > 1
			clusterassignments, M = NMFk.clustersolutions(WBig)
			if !quiet
				info("Cluster assignments:")
				display(clusterassignments)
				info("Cluster centroids:")
				display(M)
			end
			Wa, clustersilhouettes, Wv = NMFk.finalize(WBig, clusterassignments)
			minsilhouette[i] = minimum(clustersilhouettes)
			if !quiet
				info("Silhouettes for each of the $(length(clustersilhouettes)) clusters:" )
				display(clustersilhouettes')
				println("Mean silhouette = ", mean(clustersilhouettes))
				println("Min  silhouette = ", minimum(clustersilhouettes))
			end
		else
			minsilhouette[i] = NaN
		end
		imin = indmin(residues2)
		cpf[i] = cpi[imin]
		X_esta[i] = TensorDecompositions.compose(cpf[i])
		residues[i] = TensorDecompositions.rel_residue(X_esta[i], X)
		correlations[i,:] = mincorrelations(X_esta[i], X)
		println("$i - $(tranks[i]): residual $(residues[i]) worst tensor correlations $(correlations[i,:]) silhouette $(minsilhouette[i])")
	end
	info("Decompositions:")
	ibest = 1
	best = Inf
	for i in 1:nruns
		if residues[i] < best
			best = residues[i]
			ibest = i
		end
		println("$i - $(tranks[i]): residual $(residues[i]) worst tensor correlations $(correlations[i,:]) silhouette $(minsilhouette[i])")
	end
	csize = length(cpf[ibest].lambdas)
	info("Estimated true core size: $(csize)")
	JLD.save("$(resultdir)/$(prefix)$(csize).jld", "t", cpf)
	return cpf, csize, ibest
end

function getsizes(csize::Tuple, tsize::Tuple=csize .+ 1)
	ndimensons = length(tsize)
	@assert ndimensons == length(csize)
	sizes = [csize]
	for i = 1:ndimensons
		nt = ntuple(k->(k == i ? min(tsize[i], csize[i] + 1) : csize[k]), ndimensons)
		addsize = true
		for j = 1:length(sizes)
			if sizes[j] == nt
				addsize = false
				break
			end
		end
		addsize && push!(sizes, nt)
		nt = ntuple(k->(k == i ? max(1, csize[i] - 1) : csize[k]), ndimensons)
		addsize = true
		for j = 1:length(sizes)
			if sizes[j] == nt
				addsize = false
				break
			end
		end
		addsize && push!(sizes, nt)
	end
	return sizes
end

function atensor(X::Array)
	nd = ndims(X)
	info("Number of dimensions: $nd")
	tsize = size(X)
	for i = 1:nd
		info("D$i ($(tsize[i]))")
		for j = 1:tsize[i]
			st = ntuple(k->(k == i ? j : Colon()), nd)
			if nd == 3
				r = rank(X[st...])
			else
				r = TensorToolbox.mrank(X[st...])
			end
			z = count(X[st...] .> 0)
			println("$j : rank $r non-zeros $z")
			# display(X[st...])
		end
	end
end

function normalizefactors!{T,N}(X::TensorDecompositions.Tucker{T,N})
	# Xi = TensorDecompositions.compose(X)
	l = size(X.core)
	for i = 1:N
		m = maximum(X.factors[i], 1)
		@assert length(m) == l[i]
		for j = 1:l[i]
			ind = map(k->((i==k) ? j : Colon()), 1:N)
			X.core[ind...] .*= m[j]
		end
		m[m.==0] = 1.0
		X.factors[i] ./= m
	end
	# Xe = TensorDecompositions.compose(X)
	# vecnorm(Xi .- Xe)
end

function normalizefactors!{T,N}(X::TensorDecompositions.CANDECOMP{T,N})
	# Xi = TensorDecompositions.compose(X)
	for i = 1:N
		m = maximum(X.factors[i], 1)
		X.lambdas .*= vec(m)
		m[m.==0] = 1.0
		X.factors[i] ./= m
	end
	# Xe = TensorDecompositions.compose(X)
	# vecnorm(Xi .- Xe)
end

function normalizecore!{T,N}(X::TensorDecompositions.Tucker{T,N})
	# Xi = TensorDecompositions.compose(X)
	l = size(X.core)
	v = collect(1:N)
	for i = 1:N
		m = vec(maximum(X.core, v[v.!=i]))
		X.factors[i] .*= m'
		m[m.==0] = 1.0
		for j = 1:l[i]
			ind = map(k->((i==k) ? j : Colon()), 1:N)
			X.core[ind...] ./= m[j]
		end
	end
	# Xe = TensorDecompositions.compose(X)
	# vecnorm(Xi .- Xe)
end

function normalizelambdas!{T,N}(X::TensorDecompositions.CANDECOMP{T,N})
	# Xi = TensorDecompositions.compose(X)
	m = vec(X.lambdas)' .^ (1/N)
	for i = 1:N
		X.factors[i] .*= m
	end
	m = copy(X.lambdas)
	m[m.==0] = 1.0
	X.lambdas ./= m
	# Xe = TensorDecompositions.compose(X)
	# vecnorm(Xi .- Xe)
end

function mincorrelations{T,N}(X1::Array{T,N}, X2::Array{T,N})
	tsize = size(X1)
	@assert tsize == size(X2)
	c = Vector{T}(N)
	c[1] = minimum(map(j->minimum(map(k->corinf(X1[:,k,j], X2[:,k,j]), 1:tsize[2])), 1:tsize[3]))
	c[2] = minimum(map(j->minimum(map(k->corinf(X1[k,:,j], X2[k,:,j]), 1:tsize[1])), 1:tsize[3]))
	c[3] = minimum(map(j->minimum(map(k->corinf(X1[k,j,:], X2[k,j,:]), 1:tsize[1])), 1:tsize[2]))
	return c
end

function corinf{T}(v1::Vector{T}, v2::Vector{T})
	c = abs.(cor(v1, v2))
	c = isnan(c) ? Inf : c
end