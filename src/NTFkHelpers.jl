import Interpolations
import Dates
import DelimitedFiles
import DocumentFunction
import Statistics

"Convert `@sprintf` macro into `sprintf` function"
sprintf(args...) = eval(:@sprintf($(args...)))

"""
Parse files in a directory

$(DocumentFunction.documentfunction(searchdir))
"""
searchdir(key::Union{Regex,String}, path::String=".") = filter(x->occursin(key, x), readdir(path))

"""
Set image dpi

$(DocumentFunction.documentfunction(setdpi))
"""
function setdpi(dpi::Integer)
	global imagedpi = dpi
end

function setoutputformat(extension::String)
	global outputformat = extension
end

function clusterfactors(W, quiet=true)
	clusterassignments, M = NMFk.clustersolutions(W)
	if !quiet
		@info("Cluster assignments:")
		display(clusterassignments)
		@info("Cluster centroids:")
		display(M)
	end
	_, clustersilhouettes, _ = NMFk.finalize(W, clusterassignments)
	if !quiet
		@info("Silhouettes for each of the $(length(clustersilhouettes)) clusters:" )
		display(clustersilhouettes')
		println("Mean silhouette = ", Statistics.mean(clustersilhouettes))
		println("Min  silhouette = ", minimum(clustersilhouettes))
	end
	return minimum(clustersilhouettes)
end

function mincorrelations(X1::AbstractArray{T,N}, X2::AbstractArray{T,N}) where {T,N}
	c = Vector{T}(undef, N)
	if N == 3
		tsize = size(X1)
		@assert tsize == size(X2)
		c[1] = minimum(map(j->minimum(map(k->corinf(X1[:,k,j], X2[:,k,j]), 1:tsize[2])), 1:tsize[3]))
		c[2] = minimum(map(j->minimum(map(k->corinf(X1[k,:,j], X2[k,:,j]), 1:tsize[1])), 1:tsize[3]))
		c[3] = minimum(map(j->minimum(map(k->corinf(X1[k,j,:], X2[k,j,:]), 1:tsize[1])), 1:tsize[2]))
		return c
	else
		@warn("Minimum correlations can be computed for 3 dimensional tensors only; D=$N")
		c .= NaN
		return c
	end
end

function corinf(v1::Vector{T}, v2::Vector{T}) where {T}
	c = abs.(Statistics.cor(v1, v2))
	c = isnan(c) ? Inf : c
end

function flatten(X::AbstractArray{T,N}, mask::BitArray{M}) where {T,N,M}
	@assert N - 1 == M
	sz = size(X)
	A = Array{T}(undef, sum(.!mask), sz[end])
	for i = 1:sz[end]
		nt = ntuple(k->(k == N ? i : Colon()), N)
		A[:, i] = X[nt...][.!mask]
	end
	return A
end

function flatten(X::AbstractArray{T,N}, dim::Number=1) where {T,N}
	sz = size(X)
	nt = Vector{Int64}(undef, 0)
	for k = 1:N
		if (k != dim)
			push!(nt, k)
		end
	end
	A = Array{T}(undef, *(sz[nt]...), sz[dim])
	for i = 1:sz[dim]
		nt = ntuple(k->(k == dim ? i : Colon()), N)
		A[:, i] = vec(X[nt...])
	end
	return A
end

function indicize(v; rev=false, nbins=length(v), minvalue=minimum(v), maxvalue=maximum(v), stepvalue=nothing)
	if stepvalue != nothing
		if typeof(minvalue) <: Dates.DateTime
			maxvalue = ceil(maxvalue, stepvalue)
			minvalue = floor(minvalue, stepvalue)
			nbins = convert(Int, (maxvalue - minvalue) / convert(Dates.Millisecond, stepvalue))
		elseif typeof(minvalue) <: Dates.Date
			maxvalue = ceil(maxvalue, stepvalue)
			minvalue = floor(minvalue, stepvalue)
			nbins = convert(Int, (maxvalue - minvalue) / Core.eval(Main, Meta.parse(stepvalue))(1))
		else
			granularity = -convert(Int, ceil(log10(stepvalue)))
			maxvalue = ceil(maxvalue, granularity)
			minvalue = floor(minvalue, granularity)
			nbins = convert(Int, ceil.((maxvalue - minvalue) / float(stepvalue)))
		end
	end
	iv = convert(Vector{Int64}, ceil.((v .- minvalue) ./ (maxvalue - minvalue) .* nbins))
	iv[iv .== 0] .= 1
	if rev == true
		iv = (nbins + 1) .- iv
	end
	return iv
end

function bincoordinates(v; rev=false, nbins=length(v), minvalue=minimum(v), maxvalue=maximum(v), stepvalue=nothing)
	if stepvalue != nothing
		if typeof(minvalue) <: Dates.DateTime
			maxvalue = ceil(maxvalue, stepvalue)
			minvalue = floor(minvalue, stepvalue)
			nbins = convert(Int, (maxvalue - minvalue) / convert(Dates.Millisecond, stepvalue))
		elseif typeof(minvalue) <: Dates.Date
			maxvalue = ceil(maxvalue, stepvalue)
			minvalue = floor(minvalue, stepvalue)
			nbins = convert(Int, (maxvalue - minvalue) / Core.eval(Main, Meta.parse(stepvalue))(1))
		else
			granularity = -convert(Int, ceil(log10(stepvalue)))
			maxvalue = ceil(maxvalue, granularity)
			minvalue = floor(minvalue, granularity)
			nbins = convert(Int, ceil.((maxvalue - minvalue) / float(stepvalue)))
		end
	end
	if typeof(minvalue) <: Dates.DateTime || typeof(minvalue) <: Dates.Date
		stepv = (maxvalue - minvalue) / float(nbins)
		halfstepv = stepv / float(2)
		vs = collect(Base.range(minvalue + halfstepv, maxvalue - halfstepv; step=stepv))
	else
		halfstepv = (maxvalue - minvalue) / (2 * nbins)
		vs = collect(Base.range(minvalue + halfstepv, maxvalue - halfstepv; length=nbins))
	end
	if rev == true
		vs = reverse(vs)
	end
	return vs
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

function getcsize(case::String; resultdir::String=".", longname=false, extension=outputformat)
	files = searchdir(case, resultdir)
	csize = Vector{Vector{Int64}}(undef, 0)
	kwa = Vector{String}(undef, 0)
	for (i, f) in enumerate(files)
		if longname
			m = match(Regex(string("$(case)(.*)-[0-9]+_[0-9]+_[0-9]+->([0-9]+)_([0-9]+)_([0-9]+).$extension\$")), f)
		else
			m = match(Regex(string("$(case)(.*)-([0-9]+)_([0-9]+)_([0-9]+).$extension\$")), f)
		end
		if m != nothing
			push!(kwa, m.captures[1])
			c = parse.(Int64, m.captures[2:end])
			push!(csize, c)
		end
	end
	return csize, kwa
end

function getfactor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1, cutoff::Number=0)
	i = vec(maximum(t.factors[dim]; dims=1) .> cutoff)
	s = size(t.factors[dim])
	println("Factor $dim: size $s -> ($(s[1]), $(sum(i)))")
	t.factors[dim][:, i]
end

function gettensorcomponentsold(t::TensorDecompositions.Tucker, dim::Integer=1; core::Bool=false)
	cs = size(t.core)[dim]
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	Xe = Vector{Any}(undef, cs)
	tt = deepcopy(t)
	for i = 1:cs
		if core
			for j = 1:cs
				if i !== j
					nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
					tt.core[nt...] .= 0
				end
			end
			Xe[i] = TensorDecompositions.compose(tt)
			tt.core .= t.core
		else
			for j = 1:cs
				if i !== j
					tt.factors[dim][:, j] .= 0
				end
			end
			Xe[i] = TensorDecompositions.compose(tt)
			tt.factors[dim] .= t.factors[dim]
		end
	end
	m = maximum.(Xe)
	imax = sortperm(m; rev=true)
	return Xe[imax[1:crank]]
end

function getgridvalues(v::Vector, d::Integer)
	l = length(v)
	Interpolations.interpolate((1:l,), v, Interpolations.Gridded(Interpolations.Linear())).(1:l/(d+1):l)
end

function getgridvalues(v::AbstractVector, r; logtransform=true)
	lv = length(v)
	lr = length(r)
	@assert lv == lr
	f = similar(v)
	try
		if logtransform
			f = Interpolations.interpolate((log10.(r[i]),), 1:length(r[i]), Interpolations.Gridded(Interpolations.Linear())).(log10.(v))
		else
			f = Interpolations.interpolate((r[i],), 1:length(r[i]), Interpolations.Gridded(Interpolations.Linear())).(v)
		end
	catch
		if logtransform
			f = Interpolations.interpolate((sort!(log10.(r[i])),), length(r[i]):-1:1, Interpolations.Gridded(Interpolations.Linear())).(log10.(v))
		else
			f = Interpolations.interpolate((sort!(r[i]),), length(r[i]):-1:1, Interpolations.Gridded(Interpolations.Linear())).(v)
		end
	end
	return f
end

function remap(v::AbstractVector, vi::UnitRange, ve::UnitRange; nonneg::Bool=true, sp=[Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line(Interpolations.OnGrid())))], ep=[Interpolations.Line(Interpolations.OnGrid())])
	lv = length(v)
	li = length(vi)
	@assert lv == li
	f1 = Vector{Float64}(undef, length(vi))
	isn = .!isnan.(v)
	itp = Interpolations.interpolate((vi[isn],), v[isn], Interpolations.Gridded(Interpolations.Linear()))
	etp = Interpolations.extrapolate(itp, ep...)
	f1 = etp(ve)
	nonneg && (f1[f1.<0] .= 0)
	# f2 = Vector{Float64}(undef, length(ve))
	# itp = Interpolations.interpolate(f1, sp...)
	# etp = Interpolations.extrapolate(itp, ep...)
	# f2 = etp(ve)
	return f1
end

function remap(v::AbstractVector, vi::AbstractVector, ve::AbstractVector; nonneg::Bool=true, sp=[Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line(Interpolations.OnGrid())))], ep=[Interpolations.Line(Interpolations.OnGrid())])
	lv = length(v)
	li = length(vi)
	@assert lv == li
	f1 = Vector{Float64}(undef, length(vi))
	isn = .!isnan.(v)
	itp = Interpolations.interpolate((vi[isn],), v[isn], Interpolations.Gridded(Interpolations.Linear()))
	etp = Interpolations.extrapolate(itp, ep...)
	f1 = etp.(ve)
	nonneg && (f1[f1.<0] .= 0)
	# f2 = Vector{Float64}(undef, length(ve))
	# itp = Interpolations.interpolate(f1, sp...)
	# etp = Interpolations.extrapolate(itp, ep...)
	# f2 = etp.(ve)
	return f1
end

function getinterpolatedtensor(t::TensorDecompositions.Tucker{T,N}, v; sp=[Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line())), Interpolations.OnGrid()]) where {T,N}
	lv = length(v)
	f = Vector(lv)
	factors = []
	for i = 1:N
		push!(factors, t.factors[i])
	end
	for j = 1:lv
		if !isnan(v[j])
			cv = size(t.factors[j], 2)
			f = Array{T}(undef, 1,cv)
			for i = 1:cv
				f[1, i] = Interpolations.interpolate(t.factors[j][:, i], sp...)[v[j]]
			end
			factors[j] = f
		end
	end
	tn = TensorDecompositions.Tucker((factors...,), t.core)
	return tn
end

function getpredictions(t::TensorDecompositions.Tucker{T,N}, dim, v; sp=[Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line(Interpolations.OnGrid())))], ep=[Interpolations.Line(Interpolations.OnGrid())]) where {T,N}
	factors = []
	for i = 1:N
		push!(factors, t.factors[i])
	end
	for j = dim
		f = Array{T}(undef, length(v), size(factors[j], 2))
		for i = 1:size(factors[j], 2)
			itp = Interpolations.interpolate(t.factors[j][:, i], sp...)
			etp = Interpolations.extrapolate(itp, ep...)
			f[:, i] = etp.(collect(v))
		end
		factors[j] = f
	end
	tn = TensorDecompositions.Tucker((factors...,), t.core)
	return tn
end

function gettensorcomponentorder(t::TensorDecompositions.Tucker, dim::Integer=1; method::Symbol=:core, firstpeak::Bool=true, flipdim::Bool=true, quiet::Bool=true)
	cs = size(t.core)[dim]
	!quiet && @info("Core size: $(size(t.core))")
	csize = TensorToolbox.mrank(t.core)
	!quiet && @info("Core mrank: $csize")
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = cs
	if method == :factormagnitude
		fmin = vec(minimum(t.factors[dim], dims=1))
		fmax = vec(maximum(t.factors[dim], dims=1))
		@assert cs == length(fmax)
		fdx = fmax .- fmin
		for i = 1:cs
			if fmax[i] == 0
				@warn("Maximum of component $i is equal to zero!")
			end
			if fdx[i] == 0
				@warn("Component $i has zero variability!")
				crank -= 1
			end
		end
		if flipdim
			ifdx = sortperm(fdx; rev=true)[1:crank]
		else
			ifdx = reverse(sortperm(fdx; rev=true)[1:crank]; dims=1)
		end
		!quiet && @info("Factor magnitudes (max - min): $fdx")
		if firstpeak
			imax = map(i->argmax(t.factors[dim][:, ifdx[i]]), 1:crank)
			order = ifdx[sortperm(imax)]
		else
			order = ifdx
		end
	else
		maxXe = Vector{Float64}(undef, cs)
		tt = deepcopy(t)
		for i = 1:cs
			if method == :core
				for j = 1:cs
					if i !== j
						nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
						tt.core[nt...] .= 0
					end
				end
				Te = TensorDecompositions.compose(tt)
				maxXe[i] = maximum(Te) - minimum(Te)
				tt.core .= t.core
			else
				for j = 1:cs
					if i !== j
						tt.factors[dim][:, j] .= 0
					end
				end
				Te = TensorDecompositions.compose(tt)
				maxXe[i] = maximum(Te) - minimum(Te)
				tt.factors[dim] .= t.factors[dim]
			end
		end
		!quiet && @info("Max core magnitudes: $maxXe")
		imax = sortperm(maxXe; rev=flipdim)
		order = imax[1:cs]
	end
	return order
end

function gettensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, prefix::String="", mask=nothing, transform=nothing, filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude), maxcomponent::Bool=false, savetensorslices::Bool=false)
	cs = size(t.core)
	ndimensons = length(cs)
	@assert dim >= 1 && dim <= ndimensons
	dimname = namedimension(ndimensons)
	cdim = cs[dim]
	if maxcomponent
		factors = []
		for i = 1:ndimensons
			if i == dim
				push!(factors, maximum(t.factors[i]; dims=1))
			else
				push!(factors, t.factors[i])
			end
		end
		tt = deepcopy(TensorDecompositions.Tucker((factors...,), t.core))
	else
		tt = deepcopy(t)
	end
	X = Vector{Any}(undef, cdim)
	@info("Computing $cdim tensor components ...")
	for i = 1:cdim
		for j = 1:cdim
			if i !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		if length(filter) == 0
			X[i] = TensorDecompositions.compose(tt)
		else
			X[i] = TensorDecompositions.compose(tt)[filter...]
		end
		if transform != nothing
			X[i] = transform.(X[i])
		end
		nanmask!(X[i], mask)
		tt.core .= t.core
	end
	if savetensorslices
		pt = getptdimensions(pdim, ndimensons)
		nt = ntuple(k->(k == dim ? 1 : Colon()), ndimensons)
		sz = size(X[1][nt...])
		NTFk.savetensorslices(X, pt, sz, order, prefix)
	end
	return X
end

function gettensorslices(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, prefix::String="", transform=nothing, mask=nothing, filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude))
	cs = size(t.core)
	ndimensons = length(cs)
	@assert dim >= 1 && dim <= ndimensons

	X = gettensorcomponents(t, dim, pdim; transpose=transpose, prefix=prefix, mask=mask, transform=transform, filter=filter, order=order, maxcomponent=true)
	pt = getptdimensions(pdim, ndimensons)
	nt = ntuple(k->(k == dim ? 1 : Colon()), ndimensons)
	sz = size(X[1][nt...])
	Xs = Vector{Array{Float64,2}}(undef, 0)
	for (i, e) in enumerate(order)
		push!(Xs, reshape(permutedims(X[e], pt), sz)[:,:])
	end
	return Xs
end

function savetensorslices(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, prefix::String="", transform=nothing, mask=nothing, filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude))
	cs = size(t.core)
	ndimensons = length(cs)
	@assert dim >= 1 && dim <= ndimensons

	X = gettensorcomponents(t, dim, pdim; transpose=transpose, prefix=prefix, mask=mask, transform=transform, filter=filter, order=order, maxcomponent=true)
	pt = getptdimensions(pdim, ndimensons)
	nt = ntuple(k->(k == dim ? 1 : Colon()), ndimensons)
	sz = size(X[1][nt...])
	NTFk.savetensorslices(X, pt, sz, order, prefix)
end

function savetensorslices(X::AbstractArray, pt, sz, order, prefix::String="")
	recursivemkdir(prefix)
	for (i, e) in enumerate(order)
		ii = lpad("$i", 4, "0")
		# @show NMFk.minimumnan(X[e]), NMFk.maximumnan(X[e])
		DelimitedFiles.writedlm("$prefix-tensorslice$ii.dat", reshape(permutedims(X[e], pt), sz)[:,:])
		# FileIO.save("$prefix-tensorslice$ii.$(outputformat)", "X", permutedims(X[order[e]], pt))
	end
end

function mrank(t::TensorDecompositions.Tucker)
   TensorToolbox.mrank(t.core)
end

function gettensorminmax(t::TensorDecompositions.Tucker, dim::Integer=1; method::Symbol=:core)
	cs = size(t.core)[dim]
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	if method == :factormagnitude
		fmin = vec(minimum(t.factors[dim]; dims=1))
		fmax = vec(maximum(t.factors[dim]; dims=1))
		@assert cs == length(fmax)
		for i = 1:cs
			if fmax[i] == 0
				@warn("Maximum of component $i is equal to zero!")
			end
		end
		@info("Max factor magnitudes: $fmax")
		@info("Min factor magnitudes: $fmin")
	elseif method == :all
		Te = TensorDecompositions.compose(t)
		tsize = size(Te)
		ts = tsize[dim]
		maxTe = Vector{Float64}(undef, ts)
		minTe = Vector{Float64}(undef, ts)
		for i = 1:tsize[dim]
			nt = ntuple(k->(k == dim ? i : Colon()), ndimensons)
			minTe[i] = minimum(Te[nt...])
			maxTe[i] = maximum(Te[nt...])
		end
		@info("Max all magnitudes: $maxTe")
		@info("Min all magnitudes: $minTe")
	else
		maxXe = Vector{Float64}(undef, cs)
		minXe = Vector{Float64}(undef, cs)
		tt = deepcopy(t)
		for i = 1:cs
			if method == :core
				for j = 1:cs
					if i !== j
						nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
						tt.core[nt...] .= 0
					end
				end
				Te = TensorDecompositions.compose(tt)
				minXe[i] = minimum(Te)
				maxXe[i] = maximum(Te)
				tt.core .= t.core
			else
				for j = 1:cs
					if i !== j
						tt.factors[dim][:, j] .= 0
					end
				end
				Te = TensorDecompositions.compose(tt)
				minXe[i] = minimum(Te)
				maxXe[i] = maximum(Te)
				tt.factors[dim] .= t.factors[dim]
			end
		end
		@info("Max core magnitudes: $maxXe")
		@info("Min core magnitudes: $minXe")
	end
end

function gettensorcomponentgroups(t::TensorDecompositions.Tucker, dim::Integer=1; cutvalue::Number=0.9)
	g = zeros(t.factors[dim][:, 1])
	v = maximum(t.factors[dim]; dims=1) .> cutvalue
	gi = 0
	for i = 1:length(v)
		if v[i]
			m = t.factors[dim][:, i] .> cutvalue
			gi += 1
			g[m] = gi
		end
	end
	@info("Number of component groups in dimension $dim is $(gi)")
	return g
end

function gettensormaximums(t::TensorDecompositions.Tucker{T,N}) where {T,N}
	for i=1:N
		v = maximum(t.factors[i]; dims=1)
		if length(v) > 10
			vv = "[$(v[1]), $(v[2]), $(v[3]), ..., $(v[end])]"
		else
			vv = v
		end
		@info("D$i factor: $(vv) Max: $(maximum(v))")
	end
	for i=1:N
		dp = Vector{Int64}(undef, 0)
		for j = 1:N
			if j != i
				push!(dp, j)
			end
		end
		v = vec(maximum(t.core; dims=dp))
		if length(v) > 10
			vv = "[$(v[1]), $(v[2]), $(v[3]), ..., $(v[end])]"
		else
			vv = v
		end
		@info("D$i core slice: $(vv) Max: $(maximum(v))")
	end
end

function recursivemkdir(s::String; filename=true, quiet=true)
	d = Vector{String}(undef, 0)
	sc = deepcopy(s)
	if !filename && sc!= ""
		push!(d, sc)
	end
	while true
		sd = splitdir(sc)
		sc = sd[1]
		if sc == ""
			break;
		end
		push!(d, sc)
	end
	for i = length(d):-1:1
		sc = d[i]
		if isfile(sc)
			@warn("File $(sc) exists! Something wrong with the path $s")
			return
		elseif !isdir(sc)
			mkdir(sc)
			!quiet && @info("Make dir $(sc)")
		else
			!quiet && @warn("Dir $(sc) exists!")
		end
	end
end

function recursivermdir(s::String; filename=true)
	d = Vector{String}(undef, )
	sc = deepcopy(s)
	if !filename && sc!= ""
		push!(d, sc)
	end
	while true
		sd = splitdir(sc)
		sc = sd[1]
		if sc == ""
			break;
		end
		push!(d, sc)
	end
	for i = 1:length(d)
		sc = d[i]
		if isdir(sc)
			rm(sc; force=true)
		end
	end
end

function nanmask!(X::Array, mask::Union{Nothing,Number})
	if mask != nothing
		X[X .<= mask] .= NaN
	end
	return nothing
end

function nanmask!(X::Array, mask::BitArray{N}, dim) where {N}
	if length(size(mask)) == length(size(X))
		X[mask] .= NaN
	else
		X[remask(mask, size(X, dim))] .= NaN
	end
	return nothing
end

function nanmask!(X::Array, mask::BitArray{N}) where {N}
	msize = vec(collect(size(mask)))
	xsize = vec(collect(size(X)))
	if length(msize) == length(xsize)
		X[mask] .= NaN
	else
		X[remask(mask, xsize[3:end])] .= NaN
	end
	return nothing
end

function remask(sm, repeats::Integer=1)
	return reshape(repeat(sm, 1, repeats), (size(sm)..., repeats))
end

function remask(sm, repeats::Tuple)
	return reshape(repeat(sm, 1, *(repeats...)), (size(sm)..., repeats...))
end

function remask(sm, repeats::Vector{Int64})
	return reshape(repeat(sm, 1, *(repeats...)), (size(sm)..., repeats...))
end

function mapsize(csize)
	c = length(csize)
	s = ""
	for i = 1:c
		if i == c
			s *= "$(csize[i])"
		else
			s *= "$(csize[i])_"
		end
	end
	return s
end

function getptdimensions(pdim::Integer, ndimensons::Integer, transpose::Bool=false)
	pt = Vector{Int64}(undef, 0)
	push!(pt, pdim)
	if transpose
		for i = ndimensons:-1:1
			if i != pdim
				push!(pt, i)
			end
		end
	else
		for i = 1:ndimensons
			if i != pdim
				push!(pt, i)
			end
		end
	end
	return pt
end

function getptdimensions(pdim::Tuple, ndimensons::Integer, transpose::Bool=false)
	return pdim
end

function checkdimension(dim::Integer, ndimensons::Integer)
	if dim > ndimensons || dim < 1
		@warn("Dimension should be >=1 or <=$(ndimensons)")
		return false
	end
	return true
end

function checkdimension(dim::Tuple, ndimensons::Integer)
	for i in dim
		if i > ndimensons || i < 1
			@warn("Dimension should be >=1 or <=$(ndimensons)")
			return false
		end
	end
	return true
end

function zerotensorcomponents!(t::TensorDecompositions.Tucker, dim::Int)
	ndimensons = length(size(t.core))
	nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
	t.core[nt...] .= 0
end

function zerotensorcomponents!(t::TensorDecompositions.CANDECOMP, dim::Int)
	ndimensons = length(size(t.core))
	nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
	t.lambdas[nt...] .= 0
end

function namedimension(ndimensons::Int; char="C", names=("T", "X", "Y"))
	if ndimensons <= 3
		dimname = names
	else
		dimname = ntuple(i->"$char$i", ndimensons)
	end
	return dimname
end

function setnewfilename(filename::String, frame::Integer=0; keyword::String="frame")
	dir = dirname(filename)
	fn = splitdir(filename)[end]
	fs = split(fn, ".")
	if length(fs) == 1
		root = fs[1]
		ext = ""
	else
		root = join(fs[1:end-1], ".")
		ext = fs[end]
	end
	if ext == ""
		ext = "png"
		fn = fn * "." * ext
	end
	if !occursin(keyword, fn)
		fn = root * "-$(keyword)000000." * ext
	end
	rtest = occursin(Regex(string("-", keyword, "[0-9]*[.].*\$")), fn)
	if rtest
		rm = match(Regex(string("-", keyword, "([0-9]*)[.](.*)\$")), fn)
		if frame == 0
			v = parse(Int, rm.captures[1]) + 1
		else
			v = frame

		end
		l = length(rm.captures[1])
		f = "%0" * string(l) * "d"
		filename = "$(fn[1:rm.offset-1])-$(keyword)$(sprintf(f, v)).$(rm.captures[2])"
		return joinpath(dir, filename)
	else
		@warn("setnewfilename failed!")
		return ""
	end
end

function getradialmap(X::Matrix, x0, y0, nr, na)
	m, n = size(X)
	itp = Interpolations.interpolate((1:m, 1:n,), X, Interpolations.Gridded(Interpolations.Constant()))
	R = Array{Float64}(undef, nr, na)
	thetadx = pi / na
	for i=1:nr
		theta = 0
		for j=1:na
			theta += thetadx
			x = x0 + i * cos(theta)
			y = y0 + i * sin(theta)
			R[i,j] = itp(x,y)
		end
	end
	return R
end

function makemovie(; movieformat="mp4", movieopacity::Bool=false, moviedir=".", prefix::String="", keyword="frame", imgformat = "png", cleanup::Bool=true, quiet::Bool=false, vspeed::Number=1.0, numberofdigits::Integer=6)
	p = joinpath(moviedir, prefix)
	if moviedir == "."
		moviedir, prefix = splitdir(prefix)
		if moviedir == ""
			moviedir = "."
		end
	end
	if movieopacity
		s = splitdir(p)
		files = searchdir(Regex(string("$(s[2])-$(keyword)", ".*\\.", imgformat)), s[1])
		for f in files
			e = splitext(f)
			c = `convert -background black -flatten -format jpg $(joinpath(s[1], f)) $(joinpath(s[1], e[1])).jpg`
			if quiet
				run(pipeline(c, stdout=devnull, stderr=devnull))
			else
				run(c)
			end
		end
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*".$imgformat -delete`)
		imgformat = "jpg"
	end
	# c = `ffmpeg -i $p-$(keyword)%06d.png -vcodec png -pix_fmt rgba -f mp4 -filter:v "setpts=$vspeed*PTS" -y $p.mp4`
	if movieformat == "avi"
		c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -vcodec png -filter:v "setpts=$vspeed*PTS" -y $p.avi`
	elseif movieformat == "webm"
		c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -vcodec libvpx -pix_fmt yuva420p -auto-alt-ref 0 -filter:v "setpts=$vspeed*PTS" -y $p.webm`
	elseif movieformat == "gif"
		c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -f gif -filter:v "setpts=$vspeed*PTS" -y $p.gif`
	elseif movieformat == "mp4"
		c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 -filter:v "setpts=$vspeed*PTS" -y $p.mp4`
	else
		@warn("Unknown movie format $movieformat; mp4 will be used!")
		movieformat = "mp4"
		c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 -filter:v "setpts=$vspeed*PTS" -y $p.mp4`
	end
	if quiet
		run(pipeline(c, stdout=devnull, stderr=devnull))
	else
		run(c)
	end
	cleanup && run(`find $moviedir -name $prefix-$(keyword)"*".$imgformat -delete`)
	return "$p.$movieformat"
end

function movievstack(movies...; vspeed::Number=1.0, newname="seismicity"=>"all")
	nm = length(movies)
	moviesall = nothing
	for m = 1:nm
		if occursin(newname[1], movies[m][1])
			moviesall = map(i->replace(i, newname), movies[m])
			break
		end
	end
	if moviesall != nothing
		for i = 1:length(movies[1])
			c = "ffmpeg"
			v = ""
			z = ""
			for m = 1:nm
				c *= " -i $(movies[m][i])"
				v *= "[$(m-1):v]setpts=$(vspeed)*PTS[v$m];"
				z *= "[v$m]"
			end
			@show v
			c *= " -filter_complex \"$(v) $(z)vstack=inputs=$(nm)[v]\" -map \"[v]\" $(moviesall[i])"
			@show c
			run(`bash -c $c`)
		end
		return moviesall
	else
		@warn("Movie filenames cannot be renamed $(newname)!")
	end
end

function float2date(f::AbstractFloat, period::Type{<:Dates.Period}=Dates.Nanosecond)
	integer, reminder = divrem(f, 1)
	year_start = Dates.DateTime(integer)
	year = period((year_start + Dates.Year(1)) - year_start)
	partial = period(round(Dates.value(year) * reminder))
	return year_start + partial
end

function movingwindow(A::AbstractArray{T, N}, windowsize::Number=1; functionname::String="maximum") where {T, N}
	if windowsize == 0
		return A
	end
	B = similar(A)
	R = CartesianIndices(size(A))
	Istart, Iend = first(R), last(R)
	for I in R
		s = Vector{T}(undef, 0)
		a = max(Istart, I - windowsize * one(I))
		b = min(Iend, I + windowsize * one(I))
		ci = ntuple(i->a[i]:b[i], length(a))
		for J in CartesianIndices(ci)
			push!(s, A[J])
		end
		B[I] = Core.eval(NTFk, Meta.parse(functionname))(s)
	end
	return B
end