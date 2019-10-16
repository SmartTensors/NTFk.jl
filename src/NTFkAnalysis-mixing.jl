import TensorDecompositions

function loadcase(case::String; datadir::String=".")
	f = "$(datadir)/$(case)F.jld"
	if isfile(f)
		F = JLD.load(f, "X")
	else
		@warn("File $f is missing")
		return nothing
	end
	f = "$(datadir)/$(case)G.jld"
	if isfile(f)
		G = JLD.load(f, "X")
	else
		@warn("File $f is missing")
		return nothing
	end
	# A = max.(F .- G, 0)
	# B = max.(G .- F, 0)
	C = max.(F .- max.(F .- G, 0), 0)
	return C
end

function loadresults(case::String, csize::Tuple=(); resultdir::String=".")
	filename = "$(resultdir)/$(case)-$(mapsize(csize)).jld"
	if isfile(filename)
		t = NTFk.loadtucker(filename, "t")
		return t
	else
		@warn("File $(filename) does not exist!")
		return nothing
	end
end

function analysistime1(case::String; timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, trank=10, datadir::String=".", resultdir::String=".", moviedir::String=".", figuredir::String=".", suffix::String="", seed::Number=0, max_iter=DMAXITER, tol=1e-8, ini_decomp=nothing, lambda::Number=0.1)
	recursivemkdir(resultdir; filename=false)
	recursivemkdir(moviedir; filename=false)
	recursivemkdir(figuredir; filename=false)
	C = loadcase(case; datadir=datadir)
	if C == nothing
		return (0,0,0)
	end
	csize = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, trank=trank, datadir=datadir, resultdir=resultdir, moviedir=moviedir, figuredir=figuredir, skipmakemovies=true, problemname="sparse", seed=seed, max_iter=max_iter, tol=tol, ini_decomp=ini_decomp, lambda=lambda)
	return csize
end

function analysistime(case::String; timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, trank=10, datadir::String=".", resultdir::String=".", moviedir::String=".", figuredir::String=".", suffix::String="", seed::Number=0, max_iter=DMAXITER, tol=1e-8)
	recursivemkdir(resultdir; filename=false)
	recursivemkdir(moviedir; filename=false)
	recursivemkdir(figuredir; filename=false)
	C = loadcase(case; datadir=datadir)
	if C == nothing
		return (0,0,0)
	end
	csize = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, trank=trank, datadir=datadir, resultdir=resultdir, moviedir=moviedir, figuredir=figuredir, lambda=0.1, problemname="sparse", seed=seed, max_iter=max_iter, tol=tol, ini_decomp=:hosvd)
	_ = analysis("$(case)C" * suffix, C; timeindex=timeindex, xindex=xindex, yindex=yindex, trank=csize[1], datadir=datadir, resultdir=resultdir, moviedir=moviedir, figuredir=figuredir, lambda=0.000000001, problemname="dense", seed=seed, max_iter=max_iter, tol=tol, ini_decomp=nothing)
	return csize
end

function analysis(case::String; timeindex=1:5:1000, xindex=1:1:81, yindex=1:1:81, trank1=10, trank2=3, datadir::String=".", resultdir::String=".", moviedir::String=".", suffix::String="", seed::Number=0, max_iter=DMAXITER, tol=1e-8, ini_decomp=nothing)
	recursivemkdir(resultdir; filename=false)
	recursivemkdir(moviedir; filename=false)
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
		recursivemkdir(moviedir; filename=false)
		if !skipmakemovies && !skipmakedatamovie
			@info("Making problem movie for $(case) ...")
			NTFk.plottensor(X[timeindex, xindex, yindex]; movie=makemovie, moviedir=moviedir, prefix="$(case)", quiet=quiet)
		end
		xrank = length(collect(xindex))
		yrank = length(collect(yindex))
		trank = trank
		@info("Solving $(problemname) problem for $(case) ...")
		recursivemkdir(resultdir; filename=false)
		t, csize = NTFk.analysis(X[timeindex, xindex, yindex], [(trank, xrank, yrank)]; resultdir=resultdir, prefix="$(case)-", seed=seed, tol=tol, ini_decomp=ini_decomp, core_nonneg=true, verbose=false, max_iter=max_iter, lambda=lambda)
	else
		t = loadresults(case, csize; resultdir=resultdir)
		if t == nothing
			return csize
		end
	end
	if !skipmakemovies
		if !skipmaketimemovies
			recursivemkdir(moviedir; filename=false)
			@info("Making $(problemname) problem comparison movie for $(case) ...")
			nt = TensorDecompositions.compose(t[1])
			NTFk.plotcmptensors(X[timeindex, xindex, yindex], nt; movie=makemovie, moviedir=moviedir, prefix="$(case)-$((mapsize(csize)))", quiet=quiet)
			@info("Making $(problemname) problem leftover movie for $(case) ...")
			NTFk.plotlefttensor(X[timeindex, xindex, yindex], nt, X[timeindex, xindex, yindex] .- nt; movie=makemovie, moviedir=moviedir, prefix="$(case)-$((mapsize(csize)))-left", quiet=quiet)
			@info("Making $(problemname) problem component T movie for $(case) ...")
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 1; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$((mapsize(csize)))-t", quiet=quiet)
		end
		@info("Making $(problemname) 2D component plot for $(case) ...")
		recursivemkdir(figuredir; filename=false)
		NTFk.plot2dtensorcomponents(t[1]; quiet=quiet, filename="$(case)-$((mapsize(csize)))-t2d.png", figuredir=figuredir)
		if !skipmaketimemovies && !skipxymakemovies
			@info("Making $(problemname) problem component X movie for $(case) ...")
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 2; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$((mapsize(csize)))-x", quiet=quiet)
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 2, 1; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$((mapsize(csize)))-xt", quiet=quiet)
			@info("Making $(problemname) problem component Y movie for $(case) ...")
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 3; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$((mapsize(csize)))-y", quiet=quiet)
			NTFk.plottensorcomponents(X[timeindex, xindex, yindex], t[1], 3, 1; csize=csize, movie=makemovie, moviedir=moviedir, prefix="$(case)-$((mapsize(csize)))-yt", quiet=quiet)
		end
	end
	return csize
end