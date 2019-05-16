import Gadfly
import Measures
import Compose
import TensorToolbox
import TensorDecompositions
import Statistics

function movie2dtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, M=nothing; order=gettensorcomponentorder(t, dim; method=:factormagnitude), quiet::Bool=false, timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", movie=true, prefix="", dpi::Integer=imagedpi, hsize=12Compose.inch, vsize=2Compose.inch, moviedir=".", vspeed=1.0, keyword="frame", movieformat="mp4", movieopacity::Bool=false, cleanup::Bool=true, kw...)
	nc = length(order)
	if M != nothing
		np = convert(Int, ceil(nc / 3))
		x = reshape(collect(1:3*np), (3, np))
		x[x.>nc] .= nc
	else
		np = 1
	end
	movie && prefix != "" && (moviefiles = Vector{String}(undef, np))
	for f = 1:np
		if M != nothing
			filter = vec(x[:,f])
			prefixnew = prefix == "" ? "" : prefix * "-$(join(filter, "_"))"
		else
			filter = 1:nc
			prefixnew = prefix
		end
		s = plot2dtensorcomponents(t, dim; kw..., order=order, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, code=true, filter=filter)
		nt = size(t.factors[dim], 1)
		timestep = 1 / nt
		progressbar_2d = make_progressbar_2d(s)
		if movie
			for i = 1:nt
				framename = "Time $i"
				p = progressbar_2d(i, timescale, timestep, datestart, dateend, dateincrement)
				!quiet && (println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize, dpi=dpi), p); println())
				if prefixnew != ""
					filename = setnewfilename(prefixnew, i; keyword=keyword)
					Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=dpi), p)
				end
			end
		else
			!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize, dpi=dpi), Gadfly.plot(s...)); println())
			if prefixnew != ""
				filename = prefixnew * ".png"
				Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=dpi), Gadfly.plot(s...))
			end
		end
		if movie && prefix != ""
			moviefiles[f] = makemovie(movieformat=movieformat, movieopacity=movieopacity, moviedir=moviedir, prefix=prefixnew, keyword=keyword, cleanup=cleanup, quiet=quiet, vspeed=vspeed)
		end
	end
	if movie && prefix != ""
		return moviefiles
	end
end

function plot2dtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, dpi::Integer=imagedpi, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), filter=vec(1:length(order)), xmin=datestart, xmax=dateend, xfilter=nothing, transform=nothing, linewidth=2Gadfly.pt, separate::Bool=false)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	nx, ny = size(t.factors[dim])
	if datestart != nothing
		xvalues = NTFk.daterange(datestart, nx; dateend=dateend, dateincrement=dateincrement)
	else
		if xmax == nothing
			xmax = timescale ? 1 : nx
		end
		xvalues = timescale ? vec(collect(xmax/nx:xmax/nx:xmax)) : vec(collect(1:nx))
	end
	if xfilter == nothing
		xfilter = 1:nx
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
	pl = Vector{Any}(undef, ncomponents)
	for i = 1:ncomponents
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues[xfilter], y=p[xfilter, order[filter[i]]], Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=cc))
	end
	# @show [repeat(colors, colorloops); colors[1:(ncomponents - colorloops * ncolors)]]
	# tc = loopcolors ? [Gadfly.Guide.manual_color_key("", componentnames, [repeat(colors, colorloops); colors[1:(ncomponents - colorloops * ncolors)]])] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:ncomponents])] # this does not work
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:ncomponents])]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)]
	end
	if separate
		for i = 1:ncomponents
			tt = title == "" ? title : title * ": Signal #$(order[filter[i]])"
			ff = Gadfly.plot(Gadfly.layer(x=xvalues[xfilter], y=p[xfilter, order[filter[i]]], Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=parse(Colors.Colorant, "red"))), Gadfly.Guide.title(tt), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
			!quiet && (display(ff); println())
			fs = split(filename, ".")
			fn = fs[1] * "-$(lpad("$(order[filter[i]])", 4, "0"))." * fs[2]
			Gadfly.draw(Gadfly.PNG(joinpath(figuredir, fn), hsize, vsize, dpi=dpi), ff)
		end
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=dpi), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, functionname::String="Statistics.mean"; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, dpi::Integer=imagedpi, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], linewidth=2Gadfly.pt, timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend, transform=nothing)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	if datestart != nothing
		xvalues = NTFk.daterange(datestart, nx; dateend=dateend, dateincrement=dateincrement)
		xmin = minimum(xvalues)
		xmax = maximum(xvalues)
	else
		if xmax == nothing
			xmax = 1
		end
		xvalues = timescale ? vec(collect(xmax/nx:xmax/nx:xmax)) : vec(collect(1:nx))
	end
	componentnames = map(i->"T$i", 1:crank)
	dp = Vector{Int64}(undef, 0)
	for i = 1:ndimensons
		if i != dim
			push!(dp, i)
		end
	end
	pl = Vector{Any}(undef, crank)
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
		tm = Core.eval(NTFk, Meta.parse(functionname))(X2; dims=dp)
		if transform != nothing
			tm = transform.(tm)
		end
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=cc))
	end
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:crank])]
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), tc...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=dpi), ff)
	end
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), tc...]
	else
		return ff
	end
end

function plot2dmodtensorcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, functionname1::String="Statistics.mean", functionname2::String="Statistics.mean"; quiet=false, hsize=8Compose.inch, vsize=4Compose.inch, dpi::Integer=imagedpi, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], linewidth=2Gadfly.pt, timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend, transform=nothing)
	csize = TensorToolbox.mrank(t.core)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	if datestart != nothing
		xvalues = NTFk.daterange(datestart, nx; dateend=dateend, dateincrement=dateincrement)
		xmin=minimum(xvalues)
		xmax=maximum(xvalues)
	else
		if xmax == nothing
			xmax = 1
		end
		xvalues = timescale ? vec(collect(xmax/nx:xmax/nx:xmax)) : vec(collect(1:nx))
	end
	componentnames = map(i->"T$i", 1:crank)
	dp = Vector{Int64}(undef, 0)
	for i = 1:ndimensons
		if i != dim
			push!(dp, i)
		end
	end
	pl = Vector{Any}(undef, crank+2)
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
		tm = Core.eval(NTFk, Meta.parse(functionname1))(X2; dims=dp)
		if transform != nothing
			tm = transform.(tm)
		end
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=cc))
	end
	tm = map(j->Core.eval(NTFk, Meta.parse(functionname2))(vec(X[ntuple(k->(k == dim ? j : Colon()), ndimensons)...])), 1:nx)
	pl[crank+1] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth+1Gadfly.pt, line_style=[:dot], default_color="gray"))
	Xe = TensorDecompositions.compose(t)
	tm = map(j->Core.eval(NTFk, Meta.parse(functionname2))(vec(Xe[ntuple(k->(k == dim ? j : Colon()), ndimensons)...])), 1:nx)
	pl[crank+2] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color="gray85"))
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", [componentnames; "Est."; "True"], [colors[1:crank]; "gray85"; "gray"])]
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), tc...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=dpi), ff)
	end
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), tc...]
	else
		return ff
	end
end

"""
colors=[parse(Colors.Colorant, "green"), parse(Colors.Colorant, "orange"), parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "gray")]
gm=[Gadfly.Guide.manual_color_key("", ["Oil", "Gas", "Water"], colors[1:3]), Gadfly.Theme(major_label_font_size=16Gadfly.pt, key_label_font_size=14Gadfly.pt, minor_label_font_size=12Gadfly.pt)]
"""
function plot2d(T::AbstractArray, Te::AbstractArray=T; quiet::Bool=false, wellnames=nothing, Tmax=nothing, Tmin=nothing, xtitle::String="", ytitle::String="", titletext::String="", figuredir::String="results", hsize=8Gadfly.inch, vsize=4Gadfly.inch, dpi::Integer=imagedpi, keyword::String="", dimname::String="Column", colors=NTFk.colors, linewidth::Measures.Length{:mm,Float64}=2Gadfly.pt, gm=[Gadfly.Theme(key_position=:right, background_color=nothing, major_label_font_size=16Gadfly.pt, key_label_font_size=14Gadfly.pt, minor_label_font_size=12Gadfly.pt)], xaxis=1:size(Te,2), xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, xintercept=[])
	recursivemkdir(figuredir)
	c = size(T)
	if length(c) == 2
		nlayers = 1
	else
		nlayers = c[3]
	end
	if wellnames != nothing
		@assert length(wellnames) == c[1]
	end
	@assert c == size(Te)
	@assert length(vec(collect(xaxis))) == c[2]
	if Tmax != nothing && Tmin != nothing
		@assert size(Tmax) == size(Tmin)
		@assert size(Tmax, 1) == c[1]
		@assert size(Tmax, 2) == c[3]
		append = ""
	else
		if maximum(T) <= 1. && maximum(Te) <= 1.
			append = "_normalized"
		else
			append = ""
		end
	end
	if keyword != ""
		append *= "_$(keyword)"
	end
	for w = 1:c[1]
		!quiet && (if wellnames != nothing
			println("$dimname $w : $(wellnames[w])")
		else
			println("$dimname $w")
		end)
		p = Vector{Any}(undef, nlayers * 2)
		pc = 1
		for i = 1:nlayers
			if nlayers == 1
				y = T[w,:]
				ye = Te[w,:]
			else
				y = T[w,:,i]
				ye = Te[w,:,i]
			end
			if Tmax != nothing && Tmin != nothing
				y = y * (Tmax[w,i] - Tmin[w,i]) + Tmin[w,i]
				ye = ye * (Tmax[w,i] - Tmin[w,i]) + Tmin[w,i]
			end
			p[pc] = Gadfly.layer(x=xaxis, y=y, xintercept=xintercept, Gadfly.Geom.line, Gadfly.Theme(line_width=linewidth, default_color=colors[i]), Gadfly.Geom.vline)
			pc += 1
			p[pc] = Gadfly.layer(x=xaxis, y=ye, xintercept=xintercept, Gadfly.Geom.line, Gadfly.Theme(line_style=[:dot], line_width=linewidth, default_color=colors[i]), Gadfly.Geom.vline)
			pc += 1
		end
		if wellnames != nothing
			if dimname != ""
				tm = [Gadfly.Guide.title("$dimname $(wellnames[w]) $titletext")]
				filename = "$(figuredir)/$(lowercase(dimname))_$(wellnames[w])$(append).pdf"
			else
				tm = []
				filename = "$(figuredir)/$(wellnames[w])$(append).pdf"
			end
		else
			tm = []
			if dimname != ""
				filename = "$(figuredir)/$(lowercase(dimname))$(append).pdf"
			else
				filename = "$(figuredir)/$(append[2:end]).pdf"
			end
		end
		yming = ymin
		ymaxg = ymax
		if ymin != nothing && length(ymin) > 1
			yming = ymin[w]
		end
		if ymax != nothing && length(ymax) > 1
			ymaxg = ymax[w]
		end
		f = Gadfly.plot(p..., tm..., Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=yming, ymax=ymaxg), gm...)
		# Gadfly.draw(Gadfly.PNG(filename, hsize, vsize, dpi=dpi), f)
		Gadfly.draw(Gadfly.PDF(filename, hsize, vsize), f)
		!quiet && (display(f); println())
	end
end

function daterange(datestart, nx; dateend=nothing, dateincrement::String="Dates.Day")
	if dateend == nothing
		xvalues = datestart .+ vec(collect(Core.eval(Main, Meta.parse(dateincrement))(0):Core.eval(Main, Meta.parse(dateincrement))(1):Core.eval(Main, Meta.parse(dateincrement))(nx-1)))
	else
		xvalues = datestart .+ (vec(collect(1:nx)) ./ nx .* (dateend .- datestart))
	end
end