import TensorDecompositions

function analysis(T::Array, sizes::Vector=[size(T)]; tol=1e-16, ini_decomp=:hosvd, core_nonneg=true, verbose=false, max_iter=50000, lambdas=fill(0.1, length(sizes[i]) + 1))
	tsize = size(T)
	ndimensons = length(sizes[1])
	nruns = length(sizes)
	residues = Array{Float64}(nruns)
	correlations = Array{Float64}(nruns, ndimensons)
	T_esta = Array{Array{Float64,3}}(nruns)
	tucker_spnn = Array{TensorDecompositions.Tucker{Float64,3}}(nruns)
	for i in 1:nruns
		info("Core size: $(sizes[i])")
		@time tucker_spnn[i] = TensorDecompositions.spnntucker(T, sizes[i]; tol=tol, ini_decomp=ini_decomp, core_nonneg=core_nonneg, verbose=verbose, max_iter=50000, lambdas=lambdas)
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

	csize = TensorToolbox.mrank(tucker_spnn[ibest].core)
	# dNTF.atensor(tucker_spnn[ibest].core)
	ndimensons = length(csize)
	info("Estimated true core size: $(csize)")

	# dNTF.plotcmptensor(T, T_esta[ibest], 3)

	return tucker_spnn, csize
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