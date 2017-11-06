import TensorDecompositions

function analysis(case::String; timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, datadir::String=".", resultdir::String=".", moviedir::String=".", suffix::String="", seed::Number=0, max_iter=1000, tol=1e-8)
	if !isdir(resultdir)
		mkdir(resultdir)
	end
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	f = "$(datadir)/$(case)F.jld"
	if isfile(f)
		F = JLD.load(f, "X")
	else
		warn("File $f is missing")
		return (0,0,0)
	end
	f = "$(datadir)/$(case)G.jld"
	if isfile(f)
		G = JLD.load(f, "X")
	else
		warn("File $f is missing")
		return (0,0,0)
	end
	# A = max.(F .- G, 0)
	# B = max.(G .- F, 0)
	C = max.(F .- max.(F .- G, 0), 0)
	cC = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, datadir=datadir, resultdir=resultdir, moviedir=moviedir, seed=seed, max_iter=max_iter, tol=tol)
	return cC
end

function analysis(case::String, X::Array; timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, datadir::String=".", resultdir::String=".", moviedir::String=".", seed::Number=0, max_iter=1000, tol=1e-8)
	info("Making problem movie for $(case) ...")
	dNTF.plottensor(X[timeindex, xindex, yindex]; movie=true, moviedir=moviedir, prefix="$(case)", quiet=true)

	trank = 10
	info("Solving sparse problem for $(case) ...")
	t, csize = dNTF.analysis(X[timeindex, xindex, yindex], [(trank, 81, 81)]; seed=seed, tol=tol, ini_decomp=:hosvd, core_nonneg=true, verbose=false, max_iter=max_iter, lambda=0.1)
	JLD.save("$(resultdir)/$(case)-$(csize[1])_$(csize[2])_$(csize[3]).jld", "t", t)
	info("Making sparse problem comparison movie for $(case) ...")
	dNTF.plotcmptensor(X[timeindex, xindex, yindex], t[1]; movie=true, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])", quiet=true)
	nt = TensorDecompositions.compose(t[1])
	info("Making sparse problem leftover movie for $(case) ...")
	dNTF.plotlefttensor(X[timeindex, xindex, yindex], nt, X[timeindex, xindex, yindex] .- nt; movie=true, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-left", quiet=true)
	trank = csize[1]
	info("Making sparse problem component T movie for $(case) ...")
	dNTF.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 1; csize=csize, movie=true, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])", quiet=true)
	info("Making sparse problem component X movie for $(case) ...")
	dNTF.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 2; csize=csize, movie=true, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-x", quiet=true)
	dNTF.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 2, 1; csize=csize, movie=true, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-xt", quiet=true)
	info("Making sparse problem component Y movie for $(case) ...")
	dNTF.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 3; csize=csize, movie=true, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-y", quiet=true)
	dNTF.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 3, 1; csize=csize, movie=true, moviedir=moviedir, prefix="$(case)-$(csize[1])_$(csize[2])_$(csize[3])-yt", quiet=true)
	# t = JLD.load("$(resultdir)/$(case)-$(c[1])_$(csize[2])_$(csize[3]).jld", "t")

	trank = 3
	info("Solving dense problem for $(case) ...")
	t, _ = dNTF.analysis(X[timeindex, xindex, yindex], [(trank, 81, 81)]; seed=seed, tol=tol, ini_decomp=nothing, core_nonneg=true, verbose=false, max_iter=max_iter, lambda=0.000000001)
	JLD.save("$(resultdir)/$(case)-$(trank)_81_81.jld", "t", t)
	info("Making dense problem comparison movie for $(case) ...")
	nt = TensorDecompositions.compose(t[1])
	dNTF.plotcmptensor(X[timeindex, xindex, yindex], nt; movie=true, moviedir=moviedir, prefix="$(case)-$(trank)_81_81", quiet=true)
	info("Making dense problem leftover movie for $(case) ...")
	dNTF.plotlefttensor(X[timeindex, xindex, yindex], nt, X[timeindex, xindex, yindex] .- nt; movie=true, moviedir=moviedir, prefix="$(case)-$(trank)_81_81-left", quiet=true)
	for i = 1:trank
		ntt = deepcopy(t[1])
		ntt.core[1:end .!= i,:,:] = 0
		info("Making dense problem movie T$i for $(case) ...")
		dNTF.plotcmptensor(X[timeindex, xindex, yindex], ntt; movie=true, moviedir=moviedir, prefix="$(case)-$(trank)_81_81-t$i", quiet=true)
	end
	return csize
end

function analysis(T::Array, sizes=[size(T)]; seed::Number=0, tol=1e-16, ini_decomp=:hosvd, core_nonneg=true, verbose=false, max_iter=50000, lambda::Number=0.1, lambdas=fill(lambda, length(size(T)) + 1))
	info("TensorDecompositions Tucker analysis ...")
	seed > 0 && srand(seed)
	tsize = size(T)
	ndimensons = length(tsize)
	nruns = length(sizes)
	residues = Array{Float64}(nruns)
	correlations = Array{Float64}(nruns, ndimensons)
	T_esta = Array{Array{Float64,3}}(nruns)
	tucker_spnn = Array{TensorDecompositions.Tucker{Float64,3}}(nruns)
	for i in 1:nruns
		info("Core size: $(sizes[i])")
		@time tucker_spnn[i] = TensorDecompositions.spnntucker(T, sizes[i]; tol=tol, ini_decomp=ini_decomp, core_nonneg=core_nonneg, verbose=verbose, max_iter=max_iter, lambdas=lambdas)
		T_esta[i] = TensorDecompositions.compose(tucker_spnn[i])
		residues[i] = TensorDecompositions.rel_residue(tucker_spnn[i])
		correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(T_esta[i][:,k,j], T[:,k,j]), 1:tsize[2])), 1:tsize[3]))
		correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(T_esta[i][k,:,j], T[k,:,j]), 1:tsize[1])), 1:tsize[3]))
		correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(T_esta[i][k,j,:], T[k,j,:]), 1:tsize[1])), 1:tsize[2]))
		println("$i - $(sizes[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core))")
	end
	info("Decompositions:")
	ibest = 1
	best = Inf
	for i in 1:nruns
		if residues[i] < best
			best = residues[i]
			ibest = i
		end
		println("$i - $(sizes[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core))")
	end

	# dNTF.atensor(tucker_spnn[ibest].core)
	csize = TensorToolbox.mrank(tucker_spnn[ibest].core)
	info("Estimated true core size: $(csize)")
	return tucker_spnn, csize
end

function analysis(T::Array, tranks::Vector{Int64}; seed::Number=-1, tol=1e-16, verbose=false, max_iter=50000, method=:ALS)
	info("TensorDecompositions CanDecomp analysis ...")
	seed >= 0 && srand(seed)
	tsize = size(T)
	ndimensons = length(tsize)
	nruns = length(tranks)
	residues = Array{Float64}(nruns)
	correlations = Array{Float64}(nruns, ndimensons)
	T_esta = Array{Array{Float64,3}}(nruns)
	cpf = Array{TensorDecompositions.CANDECOMP{Float64,3}}(nruns)
	for i in 1:nruns
		info("CP core rank: $(tranks[i])")
		factors_initial_guess = tuple([randn(dim, tranks[i]) for dim in tsize]...)
		@time cpf[i] = TensorDecompositions.candecomp(T, tranks[i], factors_initial_guess, verbose=verbose, compute_error=true, maxiter=max_iter, method=method)
		T_esta[i] = TensorDecompositions.compose(cpf[i])
		residues[i] = TensorDecompositions.rel_residue(cpf[i])
		correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(T_esta[i][:,k,j], T[:,k,j]), 1:tsize[2])), 1:tsize[3]))
		correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(T_esta[i][k,:,j], T[k,:,j]), 1:tsize[1])), 1:tsize[3]))
		correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(T_esta[i][k,j,:], T[k,j,:]), 1:tsize[1])), 1:tsize[2]))
		println("$i - $(tranks[i]): residual $(residues[i]) tensor correlations $(correlations[i,:])")
	end

	info("Decompositions:")
	ibest = 1
	best = Inf
	for i in 1:nruns
		if residues[i] < best
			best = residues[i]
			ibest = i
		end
		println("$i - $(tranks[i]): residual $(residues[i]) tensor correlations $(correlations[i,:])")
	end

	csize = length(cpf[ibest].lambdas)
	info("Estimated true core size: $(csize)")
	return cpf, csize
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
			st = ntuple(k->(k == i ? j : :), 3)
			r = rank(X[st...])
			z = count(X[st...] .> 0)
			println("$j : rank $r non-zeros $z")
			# display(X[st...])
		end
	end
end