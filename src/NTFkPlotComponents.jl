import Gadfly
import Measures
import Colors
import Compose
import TensorToolbox
import TensorDecompositions
import Statistics

function plottensorslices(X1::Array, t2::TensorDecompositions.CANDECOMP; prefix::AbstractString="", filter=(), kw...)
	recursivemkdir(prefix; filename=true)
	ndimensons = ndims(X1)
	crank = length(t2.lambdas)
	tt = deepcopy(t2)
	for i = 1:crank
		@info("Making component $i movie ...")
		tt.lambdas[1:end .!== i] = 0
		if length(filter) == 0
			X2 = TensorDecompositions.compose(tt)
		else
			X2 = TensorDecompositions.compose(tt)[filter...]
		end
		tt.lambdas .= t2.lambdas
		plot2tensors(X1, X2; progressbar=nothing, prefix=prefix * string(i), kw...)
	end
end

function plottensorslices(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t2.core), prefix::AbstractString="", filter=(), kw...)
	recursivemkdir(prefix; filename=true)
	ndimensons = ndims(X1)
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t2)
	for i = 1:crank
		@info("Making component $(dimname[dim])-$i movie ...")
		for j = 1:crank
			if i !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		if length(filter) == 0
			X2 = TensorDecompositions.compose(tt)
		else
			X2 = TensorDecompositions.compose(tt)[filter...]
		end
		tt.core .= t2.core
		title = pdim > 1 ? "$(dimname[dim])-$i" : ""
		plot2tensors(permutedims(X1, pt), permutedims(X2, pt), 1; progressbar=nothing, title=title, prefix=prefix * string(i), kw...)
	end
end

function plot2tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), mask=nothing, transform=nothing, prefix::AbstractString="", filter=(), order=getsignalorder(t, dim; method=:factormagnitude), kw...)
	recursivemkdir(prefix; filename=true)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t)
	X = Vector{Any}(undef, crank)
	for i = 1:crank
		@info("Making component $(dimname[dim])-$i movie ...")
		for j = 1:crank
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
		nanmask!(X[i], mask)
		if transform !== nothing
			X[i] = transform.(X[i])
		end
		tt.core .= t.core
	end
	plot2tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt), 1; prefix=prefix, kw...)
end

function plot2tensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t2.core), mask=nothing, transform=nothing, prefix::AbstractString="", filter=(), order=getsignalorder(t, dim; method=:factormagnitude), kw...)
	recursivemkdir(prefix; filename=true)
	ndimensons = ndims(X1)
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t2)
	X2 = Vector{Any}(undef, crank)
	for i = 1:crank
		@info("Making component $(dimname[dim])-$i movie ...")
		for j = 1:crank
			if i !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		if length(filter) == 0
			X2[i] = TensorDecompositions.compose(tt)
		else
			X2[i] = TensorDecompositions.compose(tt)[filter...]
		end
		if transform !== nothing
			X2[i] = transform.(X2[i])
		end
		nanmask!(X2[i], mask)
		tt.core .= t2.core
	end
	plot3tensors(permutedims(X1, pt), permutedims(X2[order[1]], pt), permutedims(X2[order[2]], pt), 1; prefix=prefix, kw...)
end

function plottensorandsomething(X::Array, something, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; minvalue=NMFk.minimumnan(X), maxvalue=NMFk.maximumnan(X), sizes=size(X), xtitle::AbstractString="Time", ytitle::AbstractString="Magnitude", timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=(datestart !== nothing) ? datestart + Core.eval(Main, Meta.parse(dateincrement))(sizes[dim]) : nothing, cleanup::Bool=true, movie::Bool=false, moviedir=".", prefix::AbstractString="", vspeed=1.0, keyword="frame", quiet::Bool=false, hsize::Measures.AbsoluteLength=6Compose.inch, vsize::Measures.AbsoluteLength=6Compose.inch, dpi::Integer=imagedpi, movieformat="mp4", movieopacity::Bool=false, barratio=2/3, kw...)
	ndimensons = length(sizes)
	recursivemkdir(moviedir; filename=false)
	recursivemkdir(prefix; filename=true)
	dimname = namedimension(ndimensons; char="D", names=("Row", "Column", "Layer"))
	progressbar_2d = NMFk.make_progressbar_2d(something)
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : Colon()), ndimensons)
		p1 = NMFk.plotmatrix(X[nt...]; minvalue=minvalue, maxvalue=maxvalue, quiet=true, plot=true, kw...)
		p2 = progressbar_2d(i, timescale, timestep, datestart, dateend, dateincrement)
		!quiet && (sizes[dim] > 1) && (println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize, dpi=dpi), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(p1)), Compose.compose(Compose.context(0, 0, 1, 1 - barratio), Gadfly.render(p2)))); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=dpi), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(p1)), Compose.compose(Compose.context(0, 0, 1, 1-barratio), Gadfly.render(p2))))
		end
	end
	if movie && prefix != ""
		makemovie(movieformat=movieformat, movieopacity=movieopacity, moviedir=moviedir, prefix=prefix, keyword=keyword, cleanup=cleanup, quiet=quiet, vspeed=vspeed)
	end
end

function plottensorandcomponents(X::AbstractArray, m::AbstractMatrix, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; sizes=size(X), xtitle::AbstractString="Time", ytitle::AbstractString="Magnitude", timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=(datestart !== nothing) ? datestart + Core.eval(Main, Meta.parse(dateincrement))(sizes[dim]) : nothing, quiet::Bool=false, functionname="Statistics.mean", transform2d=nothing, totals::Bool=false, kw...)
	if totals
		s2 = plot2dmodtensorcomponents(X, dim, functionname; xtitle=xtitle, ytitle=ytitle, datestart=datestart, dateend=dateend, dateincrement=dateincrement, timescale=timescale, quiet=true, code=true, transform=transform2d)
	else
		s2 = plot2dmodtensorcomponents(m, functionname; xtitle=xtitle, ytitle=ytitle, datestart=datestart, dateend=dateend, dateincrement=dateincrement, timescale=timescale, quiet=true, code=true, transform=transform2d)
	end
	plottensorandsomething(X, s2, dim, pdim; datestart=datestart, dateend=dateend, dateincrement=dateincrement, timescale=timescale, quiet=quiet, kw...)
end

function plottensorandcomponents(X::AbstractArray, t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; sizes=size(X),xtitle::AbstractString="Time", ytitle::AbstractString="Magnitude", timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=(datestart !== nothing) ? datestart + Core.eval(Main, Meta.parse(dateincrement))(sizes[dim]) : nothing, quiet::Bool=false, functionname="Statistics.mean", transform2d=nothing, totals::Bool=true, kw...)
	if totals
		s2 = plot2dmodtensorcomponents(X, t, dim, functionname; xtitle=xtitle, ytitle=ytitle, datestart=datestart, dateend=dateend, dateincrement=dateincrement, timescale=timescale, quiet=true, code=true, transform=transform2d)
	else
		s2 = plot2dmodtensorcomponents(t, dim, functionname; xtitle=xtitle, ytitle=ytitle, datestart=datestart, dateend=dateend, dateincrement=dateincrement, timescale=timescale, quiet=true, code=true, transform=transform2d)
	end
	plottensorandsomething(X, s2, dim, pdim; datestart=datestart, dateend=dateend, dateincrement=dateincrement, timescale=timescale, quiet=quiet, kw...)
end

function plot3slices_factors(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; xtitle::AbstractString="Time", ytitle::AbstractString="Magnitude", timescale::Bool=true, datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=nothing, functionname="Statistics.mean", order=getsignalorder(t, dim; method=:factormagnitude), filter=vec(1:length(order)), xmin=datestart, xmax=dateend, ymin=nothing, ymax=nothing, transform=nothing, transform2d=transform, key_label_font_size=12Gadfly.pt, gm=[], kw...)
	if !checkdimension(dim, ndims(t.core)) || !checkdimension(pdim, ndims(t.core))
		return
	end
	s2 = plottensorfactors(t, dim; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=true, code=true, order=order, filter=filter, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, transform=transform2d, gm=gm)
	progressbar_2d = NMFk.make_progressbar_2d(s2)
	plot3tensorcomponents(t, dim, pdim; timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=false, progressbar=progressbar_2d, hsize=12Compose.inch, vsize=6Compose.inch, order=order[filter], transform=transform, key_label_font_size=key_label_font_size, kw...)
	return nothing
end

function plotall3tensors(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; kw...)
	plotallMtensors(t, 3, dim, pdim; kw...)
end

function plotallMtensors(t::TensorDecompositions.Tucker, M::Integer, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; mask=nothing, csize::Tuple=TensorToolbox.mrank(t.core), transpose::Bool=false, xtitle::AbstractString="Time", ytitle::AbstractString="Magnitude", hsize::Measures.AbsoluteLength=12Compose.inch, vsize::Measures.AbsoluteLength=3Compose.inch, timescale::Bool=true, datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=nothing, functionname="Statistics.mean", order=getsignalorder(t, dim; method=:factormagnitude), prefix=nothing, maxcomponent::Bool=false, savetensorslices::Bool=false, transform=nothing, tensorfilter=(), gla=[], kw...)
	if !checkdimension(dim, ndims(t.core)) || !checkdimension(pdim, ndims(t.core))
		return
	end
	v = getbalancedvectors(length(order), M)
	X = gettensorcomponents(t, dim, pdim; prefix=prefix, mask=mask, transform=transform, order=order, maxcomponent=maxcomponent, savetensorslices=savetensorslices, filter=tensorfilter)
	if length(gla) > 1
		@assert length(gla) == nc
	else
		gla = Vector{Any}(undef, nc)
		for i = 1:nc
			gla = []
		end
	end
	moviefiles = Vector{Any}(undef, np)
	for filter in v
		prefixnew = prefix == "" ? "" : prefix * "-$(join(filter, "_"))"
		moviefiles[i] = plotMtensorslices(t, length(filter), dim, pdim; csize=csize, transpose=transpose, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=false, progressbar=nothing, hsize=hsize, vsize=vsize, order=order[filter], prefix=prefixnew, X=X, maxcomponent=maxcomponent, savetensorslices=savetensorslices, mask=mask, signalnames=["T$i" for i = filter], gla=gla, kw...)
	end
	if moviefiles[1] !== nothing
		return convert(Vector{String}, moviefiles)
	else
		return nothing
	end
end

function plotall3slices_factors(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; kw...)
	plotallMslices_factors(t, 3, dim, pdim; kw...)
end

function plotallMslices_factors(X::AbstractVector{Array{T,N}}, F::AbstractMatrix, M::Integer, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; mask=nothing, csize::Tuple=size(X[1]), transpose::Bool=false, xtitle::AbstractString="Time", ytitle::AbstractString="Magnitude", timescale::Bool=true, normalizeslices::Bool=false, datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=nothing, functionname="Statistics.mean", order=getsignalorder(F), xmin=datestart, xmax=dateend, ymin=nothing, ymax=nothing, prefix=nothing, maxcomponent::Bool=false, transform=nothing, transform2d=nothing, tensorfilter=(), gm=[], key_label_font_size=8Gadfly.pt, kw...) where {T <: Number, N}
	if !checkdimension(dim, N) || !checkdimension(pdim, N)
		return
	end
	v = getbalancedvectors(length(order), M)
	if normalizeslices
		m = NMFk.maximumnan.(X)
		X ./= m
		F .*= permutedims(m)
	end
	if maxcomponent
		XP = X
	else
		XP = Vector{Array{T,N}}(undef, length(X))
		for i = 1:length(X)
			XP[i] = Array{T}(undef, (size(X[i], 1), size(X[i], 2), size(F, 1)))
			for j = 1:size(F, 1)
				XP[i][:,:,j] .= X[i][:,:,1] .* F[j,i]
			end
		end
		XP = convert(Array{typeof(XP[1])}, XP)
	end
	for filter in v
		s2 = plottensorfactors(F; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, code=true, order=order, filter=filter, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, transform=transform2d, gm=gm)
		progressbar_2d = NMFk.make_progressbar_2d(s2)
		prefixnew = (prefix == "") || prefix === nothing ? "" : prefix * "-$(length(X))-$(join(filter, "_"))"
		plotMtensorslices(XP, length(filter), dim, pdim; csize=csize, transpose=transpose, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=false, progressbar=progressbar_2d, hsize=12Compose.inch, vsize=6Compose.inch, order=order[filter], prefix=prefixnew, maxcomponent=maxcomponent, mask=mask, signalnames=["T$i" for i = filter], key_label_font_size=key_label_font_size, kw...)
	end
	if normalizeslices
		F ./= permutedims(m)
	end
	return nothing
end

function plotallMslices_factors(t::TensorDecompositions.Tucker, M::Integer, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; mask=nothing, csize::Tuple=TensorToolbox.mrank(t.core), transpose::Bool=false, xtitle::AbstractString="Time", ytitle::AbstractString="Magnitude", timescale::Bool=true, normalizeslices::Bool=false, datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=nothing, functionname="Statistics.mean", order=getsignalorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend, ymin=nothing, ymax=nothing, prefix=nothing, maxcomponent::Bool=false, savetensorslices::Bool=false, transform=nothing, transform2d=nothing, tensorfilter=(), gm=[], key_label_font_size=8Gadfly.pt, kw...)
	if !checkdimension(dim, ndims(t.core)) || !checkdimension(pdim, ndims(t.core))
		return
	end
	v = getbalancedvectors(length(order), M)
	X = gettensorcomponents(t, dim, pdim; filter=tensorfilter, mask=mask, transform=transform, order=order, maxcomponent=maxcomponent, savetensorslices=savetensorslices)
	if normalizeslices
		m = NMFk.maximumnan.(X)
		X ./= m
		normalizecomponents!(t, dim, m)
	end
	for filter in v
		s2 = plottensorfactors(t, dim; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, code=true, order=order, filter=filter, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, transform=transform2d, gm=gm)
		progressbar_2d = NMFk.make_progressbar_2d(s2)
		prefixnew = (prefix == "") || prefix === nothing ? "" : prefix * "-$(length(X))-$(join(filter, "_"))"
		plotMtensorslices(t, length(filter), dim, pdim; csize=csize, transpose=transpose, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=false, progressbar=progressbar_2d, hsize=12Compose.inch, vsize=6Compose.inch, order=order[filter], prefix=prefixnew, X=X, maxcomponent=maxcomponent, savetensorslices=savetensorslices, mask=mask, signalnames=["T$i" for i = filter], key_label_font_size=key_label_font_size, kw...)
	end
	if normalizeslices
		normalizecomponents!(t, dim, 1 ./ m)
	end
end

function plot3maxtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; kw...)
	plot3tensorcomponents(t, dim, pdim; kw..., maxcomponent=true)
end

function plot3tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), prefix::AbstractString="", filter=(), mask=nothing, transform=nothing, order=getsignalorder(t, dim; method=:factormagnitude), maxcomponent::Bool=false, barratio=(maxcomponent) ? 1/2 : 1/3, savetensorslices::Bool=false, X=nothing, gla=[], kw...)
	if X === nothing
		X = gettensorcomponents(t, dim, pdim; prefix=prefix, filter=filter, mask=mask, transform=transform, order=order, maxcomponent=maxcomponent, savetensorslices=savetensorslices)
	end
	nc = length(X)
	if length(gla) > 1
		@assert length(gla) == nc
	else
		gla = Vector{Any}(undef, nc)
		for i = 1:nc
			gla[i] = []
		end
	end
	pt = getptdimensions(pdim, length(csize), transpose)
	filename = plot3tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt), permutedims(X[order[3]], pt), 1; prefix=prefix, barratio=barratio, gla=[gla[order[1]], gla[order[2]], gla[order[3]]], kw...)
	if maxcomponent && prefix != ""
		recursivemkdir(prefix; filename=true)
		mv("$prefix.png", "$prefix-max.png"; force=true)
		return
	else
		return filename
	end
	return nothing
end

function plotMtensorslices(X::AbstractVector{Array{T,N}}, M::Integer, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, csize::Tuple=size(X[1]), prefix::AbstractString="", filter=(), mask=nothing, transform=nothing, order=getsignalorder(X), maxcomponent::Bool=false, barratio=(maxcomponent) ? 1/2 : 1/3, gla=[], kw...) where {T <: Number, N}
	minvalue = NMFk.minimumnan(map(i->NMFk.minimumnan(X[i]), 1:M))
	maxvalue = NMFk.maximumnan(map(i->NMFk.maximumnan(X[i]), 1:M))
	pt = getptdimensions(pdim, length(csize), transpose)
	XM = Vector{Any}(undef, M)
	for i = 1:M
		XM[i] = permutedims(X[order[i]], pt)
	end
	XM = convert(Array{typeof(XM[1])}, XM)
	nc = length(X)
	if length(gla) > 1
		@assert length(gla) == nc
	else
		gla = Vector{Any}(undef, nc)
		for i = 1:nc
			gla[i] = []
		end
	end
	filename = plotMtensors(XM, 1; prefix=prefix, barratio=barratio, gla=[gla[order[i]] for i=1:M], mask=mask, kw...)
	if maxcomponent && prefix != ""
		mv("$prefix.png", "$prefix-max.png"; force=true)
		return nothing
	else
		return filename
	end
end

function plotMtensorslices(t::TensorDecompositions.Tucker, M::Integer, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), prefix::AbstractString="", filter=(), mask=nothing, transform=nothing, order=getsignalorder(t, dim; method=:factormagnitude), maxcomponent::Bool=false, barratio=(maxcomponent) ? 1/2 : 1/3, savetensorslices::Bool=false, X=nothing, gla=[], kw...)
	if X === nothing
		X = gettensorcomponents(t, dim, pdim; prefix=prefix, filter=filter, mask=mask, transform=transform, order=order, maxcomponent=maxcomponent, savetensorslices=savetensorslices)
	end
	minvalue = NMFk.minimumnan(map(i->NMFk.minimumnan(X[i]), 1:M))
	maxvalue = NMFk.maximumnan(map(i->NMFk.maximumnan(X[i]), 1:M))
	pt = getptdimensions(pdim, length(csize), transpose)
	XM = Vector{Any}(undef, M)
	for i = 1:M
		XM[i] = permutedims(X[order[i]], pt)
	end
	XM = convert(Array{typeof(XM[1])}, XM)
	nc = length(X)
	if length(gla) > 1
		@assert length(gla) == nc
	else
		gla = Vector{Any}(undef, nc)
		for i = 1:nc
			gla[i] = []
		end
	end
	filename = plotMtensors(XM, 1; prefix=prefix, barratio=barratio, gla=[gla[order[i]] for i=1:M], kw...)
	if maxcomponent && prefix != ""
		recursivemkdir(prefix; filename=true)
		mv("$prefix.png", "$prefix-max.png"; force=true)
		return nothing
	else
		return filename
	end
end

function plotalltensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Union{Integer,Tuple}=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), prefix::AbstractString="", filter=(), mask=nothing, transform=nothing, order=getsignalorder(t, dim; method=:factormagnitude), savetensorslices::Bool=false, quiet=false, kw...)
	X = gettensorcomponents(t, dim, pdim; prefix=prefix, filter=filter, mask=mask, transform=transform, order=order, maxcomponent=true, savetensorslices=savetensorslices)
	pt = getptdimensions(pdim, length(csize), transpose)
	mdfilter = ntuple(k->(k == 1 ? 1 : Colon()), length(csize))
	for i = 1:length(X)
		filename = prefix == "" ? "" : "$prefix-tensorslice$(lpad("$i", 4, "0")).png"
		p = NMFk.plotmatrix(permutedims(X[order[i]], pt)[mdfilter...]; filename=filename, plot=true, kw...)
		!quiet && (@info("Slice $i"); display(p); println();)
	end
	return nothing
end

function getbalancedvectors(nc, M)
	if M > nc
		@warn("M is greater than the number of tensor dimensions ($(M) > $(nc))")
	end
	np = convert(Int, floor(nc / M))
	np1 = convert(Int, ceil(nc / M))
	x = reshape(collect(1:M*np), (M, np))
	v = [x[:,i] for i=1:np]
	if np1 > np
		push!(v, collect(np * M + 1 : nc))
	end
	return v
end