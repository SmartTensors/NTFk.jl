import Gadfly
import Measures
import Colors
import Compose
import TensorToolbox
import TensorDecompositions
import Distributions

function plottensorcomponents(X1::Array, t2::TensorDecompositions.CANDECOMP; prefix::String="", filter=(), kw...)
	recursivemkdir(prefix)
	ndimensons = length(size(X1))
	crank = length(t2.lambdas)
	tt = deepcopy(t2)
	for i = 1:crank
		info("Making component $i movie ...")
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

function plottensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t2.core), prefix::String="", filter=(), kw...)
	recursivemkdir(prefix)
	ndimensons = length(size(X1))
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t2)
	for i = 1:crank
		info("Making component $(dimname[dim])-$i movie ...")
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

function plot2tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), mask=nothing, transform=nothing, prefix::String="", filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude), kw...)
	recursivemkdir(prefix)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t)
	X = Vector{Any}(crank)
	for i = 1:crank
		info("Making component $(dimname[dim])-$i movie ...")
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
		if transform != nothing
			X[i] = transform.(X[i])
		end
		tt.core .= t.core
	end
	plot2tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt), 1; prefix=prefix, kw...)
end

function plot2tensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t2.core), mask=nothing, transform=nothing, prefix::String="", filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude), kw...)
	recursivemkdir(prefix)
	ndimensons = length(size(X1))
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t2)
	X2 = Vector{Any}(crank)
	for i = 1:crank
		info("Making component $(dimname[dim])-$i movie ...")
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
		if transform != nothing
			X2[i] = transform.(X2[i])
		end
		nanmask!(X2[i], mask)
		tt.core .= t2.core
	end
	plot3tensors(permutedims(X1, pt), permutedims(X2[order[1]], pt), permutedims(X2[order[2]], pt), 1; prefix=prefix, kw...)
end

function plottensorandcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t.core), sizes=size(X), xtitle="Time", ytitle="Magnitude", timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=(datestart != nothing) ? datestart + eval(parse(dateincrement))(sizes[dim]) : nothing, dateincrement::String="Dates.Day", cleanup::Bool=true, movie::Bool=false, moviedir=".", prefix::String="", keyword="frame", title="", quiet::Bool=false, filter=(), minvalue=minimumnan(X), maxvalue=maximumnan(X), hsize=12Compose.inch, vsize=12Compose.inch, colormap=colormap_gyr, functionname="mean", vspeed=1.0, transform=nothing, transform2d=nothing, mask=nothing, kw...)
	ndimensons = length(sizes)
	if !checkdimension(dim, ndimensons) || !checkdimension(pdim, ndimensons)
		return
	end
	recursivemkdir(moviedir; filename=false)
	recursivemkdir(prefix)
	dimname = namedimension(ndimensons; char="D", names=("Row", "Column", "Layer"))
	s2 = plot2dmodtensorcomponents(X, t, dim, functionname; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, timescale=timescale, quiet=true, code=true, transform=transform2d)
	progressbar_2d = make_progressbar_2d(s2)
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : Colon()), ndimensons)
		p1 = plotmatrix(X[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap, transform=transform, mask=mask)
		p2 = progressbar_2d(i, timescale, timestep, datestart, dateend, dateincrement)
		!quiet && (sizes[dim] > 1) && (println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Gadfly.render(p1)), Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(p2)))); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Gadfly.render(p1)), Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(p2))))
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -filter:v "setpts=$vspeed*PTS" -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		if moviedir == "."
			moviedir, prefix = splitdir(prefix)
			if moviedir == ""
				moviedir = "."
			end
		end
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*".png -delete`)
	end
end

function plot3tensorsandcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; xtitle="Time", ytitle="Magnitude", timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", functionname="mean", order=gettensorcomponentorder(t, dim; method=:factormagnitude), filter=vec(1:length(order)), xmin=datestart, xmax=dateend, ymin=nothing, ymax=nothing, transform2d=nothing, kw...)
	ndimensons = length(t.factors)
	if !checkdimension(dim, ndimensons) || !checkdimension(pdim, ndimensons)
		return
	end
	s2 = plot2dtensorcomponents(t, dim; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=true, code=true, order=order, filter=filter, xmin=xmin, xmax=xmax, ymin=xmin, ymax=xmax, transform=transform2d)
	progressbar_2d = make_progressbar_2d(s2)
	plot3tensorcomponents(t, dim, pdim; timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=false, progressbar=progressbar_2d, hsize=12Compose.inch, vsize=6Compose.inch, order=order[filter], kw...)
end

function plotall3tensorsandcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; mask=nothing, csize::Tuple=TensorToolbox.mrank(t.core), transpose=false, xtitle="Time", ytitle="Magnitude", timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", functionname="mean", order=gettensorcomponentorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend, ymin=nothing, ymax=nothing, prefix=nothing, maxcomponent=true, savetensorslices=false, transform=nothing, transform2d=nothing, kw...)
	ndimensons = length(t.factors)
	if !checkdimension(dim, ndimensons) || !checkdimension(pdim, ndimensons)
		return
	end
	nc = size(t.factors[pdim], 2)
	np = convert(Int, ceil(nc / 3))
	x = reshape(collect(1:3*np), (3, np))
	x[x.>nc] .= nc
	X = gettensorcomponents(t, dim, pdim; transpose=transpose, csize=csize, prefix=prefix, mask=mask, transform=transform, order=order, maxcomponent=maxcomponent, savetensorslices=savetensorslices)
	for i = 1:np
		filter = vec(x[:,i])
		s2 = plot2dtensorcomponents(t, dim; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=true, code=true, order=order, filter=filter, xmin=xmin, xmax=xmax, ymin=xmin, ymax=xmax, transform=transform2d)
		progressbar_2d = make_progressbar_2d(s2)
		prefixnew = prefix == "" ? "" : prefix * "-$(join(filter, "_"))"
		plot3tensorcomponents(t, dim, pdim; csize=csize, transpose=transpose, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=false, progressbar=progressbar_2d, hsize=12Compose.inch, vsize=6Compose.inch, order=order[filter], prefix=prefixnew, X=X, maxcomponent=maxcomponent, savetensorslices=savetensorslices, mask=mask, kw...)
	end
end

function plot3maxtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; kw...)
	plot3tensorcomponents(t, dim, pdim; kw..., maxcomponent=true)
end

function plot3tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), prefix::String="", filter=(), mask=nothing, transform=nothing, order=gettensorcomponentorder(t, dim; method=:factormagnitude), maxcomponent::Bool=false, savetensorslices::Bool=false, X=nothing, kw...)
	if X == nothing
		X = gettensorcomponents(t, dim, pdim; transpose=transpose, csize=csize, prefix=prefix, filter=filter, mask=mask, transform=transform, order=order, maxcomponent=maxcomponent, savetensorslices=savetensorslices)
	end
	pt = getptdimensions(pdim, length(csize), transpose)
	barratio = (maxcomponent) ? 1/2 : 1/3
	plot3tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt), permutedims(X[order[3]], pt), 1; prefix=prefix, barratio=barratio, kw...)
	if maxcomponent && prefix != ""
		recursivemkdir(prefix)
		mv("$prefix-frame000001.png", "$prefix-max.png"; remove_destination=true)
	end
end

function plotalltensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), prefix::String="", filter=(), mask=nothing, transform=nothing, order=gettensorcomponentorder(t, dim; method=:factormagnitude), savetensorslices::Bool=false, quiet=false, kw...)
	X = gettensorcomponents(t, dim, pdim; transpose=transpose, csize=csize, prefix=prefix, filter=filter, mask=mask, transform=transform, order=order, maxcomponent=true, savetensorslices=savetensorslices)
	pt = getptdimensions(pdim, length(csize), transpose)
	mdfilter = ntuple(k->(k == 1 ? 1 : Colon()), length(csize))
	for i = 1:length(X)
		filename = prefix == "" ? "" : "$prefix-tensorslice$i.png"
		p = plotmatrix(permutedims(X[order[i]], pt)[mdfilter...]; filename=filename, kw...)
		!quiet && (info("Slice $i"); display(p); println();)
	end
end