import Gadfly
import Measures
import Colors
import Compose
import TensorToolbox
import TensorDecompositions
import Distributions

function plot2dtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true,  datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), filter=vec(1:length(order)), xmin=datestart, xmax=dateend, transform=nothing, linewidth=2Gadfly.pt, separate::Bool=false)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	if datestart != nothing
		if dateend == nothing
			xvalues = datestart .+ vec(collect(eval(parse(dateincrement))(0):eval(parse(dateincrement))(1):eval(parse(dateincrement))(nx-1)))
		else
			xvalues = datestart .+ (vec(collect(1:nx)) ./ nx .* (dateend .- datestart))
		end
	end
	ncomponents = length(filter)
	loopcolors = ncomponents > ncolors ? true : false
	# if loopcolors
	# 	colorloops = convert(Int64, floor(ncomponents / ncolors))
	# end
	componentnames = map(i->"T$i", filter)
	p = t.factors[dim]
	if transform != nothing
		p = transform.(p)
	end
	pl = Vector{Any}(ncomponents)
	for i = 1:ncomponents
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=p[:, order[filter[i]]], Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=cc))
	end
	tx = timescale ? [] : [Gadfly.Coord.Cartesian(xmin=minimum(xvalues), xmax=maximum(xvalues))]
	# @show [repeat(colors, colorloops); colors[1:(ncomponents - colorloops * ncolors)]]
	# tc = loopcolors ? [Gadfly.Guide.manual_color_key("", componentnames, [repeat(colors, colorloops); colors[1:(ncomponents - colorloops * ncolors)]])] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:ncomponents])] # this does not work
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:ncomponents])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	xm = (xmin == nothing && xmax == nothing) ? [] : [Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., tm..., xm...]
	end
	if separate
		for i = 1:ncomponents
			tt = title == "" ? title : title * " Signal $(order[filter[i]])"
			ff = Gadfly.plot(Gadfly.layer(x=xvalues, y=p[:, order[filter[i]]], Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=parse(Colors.Colorant, "red"))), Gadfly.Guide.title(tt), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., tx..., xm...)
			!quiet && (display(ff); println())
			fs = split(filename, ".")
			fn = fs[1] * "-$(lpad(order[filter[i]],4,0))." * fs[2]
			Gadfly.draw(Gadfly.PNG(joinpath(figuredir, fn), hsize, vsize, dpi=150), ff)
		end
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., tm..., tx..., xm...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, functionname::String="mean"; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend)
	recursivemkdir(figuredir; filename=false, transform=nothing)
	recursivemkdir(filename)
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	if datestart != nothing
		if dateend == nothing
			xvalues = datestart .+ vec(collect(eval(parse(dateincrement))(0):eval(parse(dateincrement))(1):eval(parse(dateincrement))(nx-1)))
		else
			xvalues = datestart .+ (vec(collect(1:nx)) ./nx .* (dateend .- datestart))
		end
	end
	componentnames = map(i->"T$i", 1:crank)
	dp = Vector{Int64}(0)
	for i = 1:ndimensons
		if i != dim
			push!(dp, i)
		end
	end
	pl = Vector{Any}(crank)
	tt = deepcopy(t)
	for (i, o) = enumerate(order)
		for j = 1:crank
			if o !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		X2 = TensorDecompositions.compose(tt)
		tt.core .= t.core
		tm = eval(parse(functionname))(X2, dp)
		if transform != nothing
			tm = transform.(tm)
		end
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tx = timescale ? [] : [Gadfly.Coord.Cartesian(xmin=minimum(xvalues), xmax=maximum(xvalues))]
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:crank])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	xm = (xmin == nothing && xmax == nothing) ? [] : [Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., xm..., tc...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., xm..., tc..., tx...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, functionname1::String="mean", functionname2::String="mean"; quiet=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend, transform=nothing)
	csize = TensorToolbox.mrank(t.core)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	if datestart != nothing
		if dateend == nothing
			xvalues = datestart .+ vec(collect(eval(parse(dateincrement))(0):eval(parse(dateincrement))(1):eval(parse(dateincrement))(nx-1)))
		else
			xvalues = datestart .+ (vec(collect(1:nx)) ./nx .* (dateend .- datestart))
		end
	end
	componentnames = map(i->"T$i", 1:crank)
	dp = Vector{Int64}(0)
	for i = 1:ndimensons
		if i != dim
			push!(dp, i)
		end
	end
	pl = Vector{Any}(crank+2)
	tt = deepcopy(t)
	for (i, o) = enumerate(order)
		for j = 1:crank
			if o !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		X2 = TensorDecompositions.compose(tt)
		tt.core .= t.core
		tm = eval(parse(functionname1))(X2, dp)
		if transform != nothing
			tm = transform.(tm)
		end
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tm = map(j->eval(parse(functionname2))(vec(X[ntuple(k->(k == dim ? j : Colon()), ndimensons)...])), 1:nx)
	pl[crank+1] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=3Gadfly.pt, line_style=:dot, default_color=parse(Colors.Colorant, "gray")))
	Xe = TensorDecompositions.compose(t)
	tm = map(j->eval(parse(functionname2))(vec(Xe[ntuple(k->(k == dim ? j : Colon()), ndimensons)...])), 1:nx)
	pl[crank+2] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=parse(Colors.Colorant, "gray85")))
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", [componentnames; "Est."; "True"], [colors[1:crank]; "gray85"; "gray"])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	xm = (xmin == nothing && xmax == nothing) ? [] : [Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., xm..., tc...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., xm..., tc...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end