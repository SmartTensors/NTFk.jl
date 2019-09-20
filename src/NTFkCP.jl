import TensorDecompositions
import DocumentFunction

"""
Candecomp/Parafac (CP) deconstruction: single tensor rank analysis

methods: ALS, SGSD, cp_als, cp_apr, cp_nmu, cp_opt, cp_sym, cp_wopt
"""
function analysis(X::AbstractArray{T,N}, trank::Integer, nTF=1; seed::Number=-1, tol=1e-8, verbose=false, max_iter=DMAXITER, method=:ALS, resultdir::String=".", prefix::String="$(string(method))", quiet=true, serial::Bool=false, saveall=false, kw...) where {T,N}
	if occursin("cp_", string(method))
		@info("MATLAB TensorToolbox CanDecomp analysis using $(string(method)) ...")
	elseif occursin("bcu_", string(method))
		@info("MATLAB Block-coordinate nonconvex CanDecomp analysis using $(string(method)) ...")
	else
		@info("TensorDecompositions CanDecomp analysis using $(string(method)) ...")
	end
	recursivemkdir(resultdir; filename=false)
	recursivemkdir(prefix; filename=false)
	if seed < 0
		seed = abs(rand(Int16))
		@info("Random seed: $seed")
	else
		@info("Provided seed: $seed")
	end
	Random.seed!(seed)
	tsize = size(X)
	ndimensons = length(tsize)
	@info("CP core rank: $(trank)")
	residues = Array{T}(undef, nTF)
	cpi = Array{TensorDecompositions.CANDECOMP{T,N}}(undef, nTF)
	WBig = Vector{Matrix}(undef, nTF)
	cpbest = nothing
	if nprocs() > 1 && !serial
		cpi = pmap(i->(Random.seed!(seed+i); NTFk.candecomp(X, trank; tsize=tsize, verbose=verbose, maxiter=max_iter, method=method, tol=tol, kw...)), 1:nTF)
	else
		for n = 1:nTF
			@time cpi[n] = NTFk.candecomp(X, trank; tsize=tsize, verbose=verbose, maxiter=max_iter, method=method, tol=tol, kw...)
		end
	end
	for n = 1:nTF
		residues[n] = TensorDecompositions.rel_residue(cpi[n], X)
		normalizelambdas!(cpi[n])
		f = map(k->cpi[n].factors[k]', 1:ndimensons)
		# p = NTFk.plotmatrix(cpi[n].factors[1]')
		# display(p); println()
		# p = NTFk.plotmatrix(f)
		# display(p); println()
		# @show minimum(cpi[n].lambdas), maximum(cpi[n].lambdas)
		WBig[n] = hcat(f...)
	end
	minsilhouette = nTF > 1 ? clusterfactors(WBig, quiet) : NaN
	imin = argmin(residues)
	csize = length(cpi[imin].lambdas)
	X_esta = TensorDecompositions.compose(cpi[imin])
	correlations = mincorrelations(X_esta, X)
	println("$(trank): residual $(residues[imin]) worst tensor correlations $(correlations) rank $(csize) silhouette $(minsilhouette)")
	if saveall
		recursivemkdir(resultdir; filename=false)
		recursivemkdir(prefix; filename=false)
		FileIO.save("$(resultdir)/$(prefix)-$(mapsize(csize)).$(outputformat)", "cp", cpi[imin])
	end
	return cpi[imin], residues[imin], correlations, minsilhouette
end

"""
Candecomp/Parafac (CP) deconstruction: multi tensor rank analyses

methods: ALS, SGSD, cp_als, cp_apr, cp_nmu, cp_opt, cp_sym, cp_wopt
"""
function analysis(X::AbstractArray{T,N}, tranks::Vector{Int}, nTF=1; seed::Number=-1, method=:ALS, resultdir::String=".", prefix::String="$(string(method))", serial::Bool=false, kw...) where {T,N}
	if seed < 0
		seed = abs(rand(Int16))
		@info("Random seed: $seed")
	else
		@info("Provided seed: $seed")
	end
	Random.seed!(seed)
	recursivemkdir(resultdir; filename=false)
	recursivemkdir(prefix; filename=false)
	tsize = size(X)
	ndimensons = length(tsize)
	nruns = length(tranks)
	residues = Array{T}(undef, nruns)
	correlations = Array{T}(undef, nruns, ndimensons)
	cpf = Array{TensorDecompositions.CANDECOMP{T,N}}(undef, nruns)
	minsilhouette = Array{Float64}(undef, nruns)
	if nprocs() > 1 && !serial
		r = pmap(i->(Random.seed!(seed+i); analysis(X, tranks[i], nTF; method=method, resultdir=resultdir, prefix=prefix, kw..., serial=true, quiet=true)), 1:nruns)
		cpf = map(i->(r[i][1]), 1:nruns)
		residues = map(i->(r[i][2]), 1:nruns)
		correlations = map(i->(r[i][3]), 1:nruns)
		minsilhouette = map(i->(r[i][4]), 1:nruns)
	else
		s = nprocs() > 1 ? false : true
		for i in 1:nruns
			cpf[i], residues[i], correlations[i, :], minsilhouette[i] = analysis(X, tranks[i], nTF; method=method, resultdir=resultdir, prefix=prefix, serial=s, kw...)
		end
	end
	@info("Decompositions:")
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
	@info("Estimated true core size (diagonal length): $(csize)")
	if prefix != ""
		if nruns > 1
			FileIO.save("$(resultdir)/$(prefix)-$(csize).$(outputformat)", "cp_vector", cpf)
		else
			FileIO.save("$(resultdir)/$(prefix)-$(csize).$(outputformat)", "cp", cpf[1])
		end
	end
	return cpf, csize, ibest
end

"""
Candecomp/Parafac (CP) deconstruction

$(DocumentFunction.documentfunction(candecomp))
"""
function candecomp(X::AbstractArray{T, N}, r::Integer; tsize=size(X), seed::Number=0, method::Symbol=:ALS, functionname::String=string(method), tol::Float64=1e-8, maxiter::Integer=DMAXITER, compute_error::Bool=true, verbose::Bool=false, kw...) where {T,N}
	if occursin("cp_", string(method))
		c = ttanalysis(X, r; seed=seed, functionname=functionname, maxiter=maxiter, tol=tol, kw...)
	elseif occursin("bcu_", string(method))
		c = bcuanalysis(X, r; seed=seed, functionname=split(functionname, "bcu_")[2], maxiter=maxiter, tol=tol, kw...)
	else
		factors_initial_guess = tuple([randn(T, d, r) for d in tsize]...)
		c = TensorDecompositions.candecomp(X, r, factors_initial_guess; verbose=verbose, compute_error=compute_error, maxiter=maxiter, method=method)
	end
	return c
end