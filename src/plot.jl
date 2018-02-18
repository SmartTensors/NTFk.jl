import Gadfly
import Colors
import Compose
import TensorToolbox
import TensorDecompositions
import Distributions

colors = ["red", "blue", "green", "orange", "magenta", "cyan", "brown", "pink", "lime", "navy", "maroon", "yellow", "olive", "springgreen", "teal", "coral", "lavender", "beige"]
ncolors = length(colors)

colormap_gyr = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"))]
colormap_gy = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"))]
colormap_wb = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "white"), parse(Colors.Colorant, "black"))]

searchdir(key::Regex, path::String = ".") = filter(x->ismatch(key, x), readdir(path))
searchdir(key::String, path::String = ".") = filter(x->contains(x, key), readdir(path))

function getcsize(case::String; resultdir::String=".")
	files = searchdir(case, resultdir)
	csize = Array{Int64}(0, 3)
	kwa = Vector{String}(0)
	for (i, f) in enumerate(files)
		m = match(Regex(string("$(case)(.*)-([0-9]+)_([0-9]+)_([0-9]+).jld")), f)
		if m != nothing
			push!(kwa, m.captures[1])
			c = parse.(Int64, m.captures[2:end])
			csize = vcat(csize, c')
		end
	end
	return csize, kwa
end

function gettensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1; core::Bool=false)
	cs = size(t.core)[dim]
	csize = TensorToolbox.mrank(t.core)
	crank = csize[dim]
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	Xe = Vector{Any}(cs)
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

function gettensorcomponentorder(t::TensorDecompositions.Tucker, dim::Integer=1; method::Symbol=:core)
	cs = size(t.core)[dim]
	csize = TensorToolbox.mrank(t.core)
	crank = csize[dim]
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	if method == :factormagnitude
		fmax = vec(maximum(t.factors[dim], 1))
		@assert cs == length(fmax)
		for i = 1:cs
			if fmax[i] == 0
				warn("Maximum of component $i is equal to zero!")
			end
		end
		ifmax = sortperm(fmax; rev=true)[1:crank]
		imax = map(i->indmax(t.factors[dim][:, ifmax[i]]), 1:crank)
		order = ifmax[sortperm(imax)]
	else
		maxXe = Vector{Float64}(cs)
		tt = deepcopy(t)
		for i = 1:cs
			if method == :core
				for j = 1:cs
					if i !== j
						nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
						tt.core[nt...] .= 0
					end
				end
				maxXe[i] = maximum(TensorDecompositions.compose(tt))
				tt.core .= t.core
			else
				for j = 1:cs
					if i !== j
						tt.factors[dim][:, j] .= 0
					end
				end
				maxXe[i] = maximum(TensorDecompositions.compose(tt))
				tt.factors[dim] .= t.factors[dim]
			end
		end
		imax = sortperm(maxXe; rev=true)
		order = imax[1:crank]
	end
	return order
end

function plot2dtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, code::Bool=false)
	csize = TensorToolbox.mrank(t.core)
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	componentnames = map(i->"T$i", 1:crank)
	order = gettensorcomponentorder(t, dim; method=:factormagnitudect)
	p = t.factors[dim]
	pl = Vector{Any}(crank)
	for i = 1:crank
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=abs.(p[:, order[i]]), Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:crank])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., tm...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., tm...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, functionname::String="mean"; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, code::Bool=false)
	csize = TensorToolbox.mrank(t.core)
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	componentnames = map(i->"T$i", 1:crank)
	order = gettensorcomponentorder(t, dim; method=:factormagnitudect)
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
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:crank])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., tc...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., tc...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, functionname1::String="mean", functionname2::String="mean"; quiet=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, code::Bool=false)
	csize = TensorToolbox.mrank(t.core)
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	componentnames = map(i->"T$i", 1:crank)
	order = gettensorcomponentorder(t, dim; method=:factormagnitudect)
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
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tm = map(j->eval(parse(functionname2))(vec(X[ntuple(k->(k == dim ? j : Colon()), ndimensons)...])), 1:nx)
	pl[crank+1] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=3Gadfly.pt, line_style=:dash, default_color=parse(Colors.Colorant, "gray")))
	Xe = TensorDecompositions.compose(t)
	tm = map(j->eval(parse(functionname2))(vec(Xe[ntuple(k->(k == dim ? j : Colon()), ndimensons)...])), 1:nx)
	pl[crank+2] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=parse(Colors.Colorant, "gray85")))
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", [componentnames; "Est."; "True"], [colors[1:crank]; "gray85"; "gray"])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., tc...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., tc...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plotmatrix(X::Matrix; minvalue=minimum(X), maxvalue=maximum(X), label="", title="", xlabel="", ylabel="", gm=[Gadfly.Guide.xticks(label=false, ticks=nothing), Gadfly.Guide.yticks(label=false, ticks=nothing)], masize::Int64=0, colormap=colormap_gyr, filename::String="", hsize=6Compose.inch, vsize=6Compose.inch, figuredir::String=".")
	Xp = min.(max.(movingaverage(X, masize), minvalue, 1e-32), maxvalue)
	p = Gadfly.spy(Xp, gm..., Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Guide.ColorKey(title=label), Gadfly.Scale.ContinuousColorScale(colormap..., minvalue=minvalue, maxvalue=maxvalue), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt))
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), p)
	end
	return p
end

function plotfactor(t::TensorDecompositions.Tucker, dim::Integer=1, cuttoff::Number=0; kw...)
	i = vec(maximum(t.factors[dim], 1) .> cuttoff)
	s = size(t.factors[dim])
	println("Factor $dim: size $s -> ($(s[1]), $(sum(i)))")
	plotmatrix(t.factors[dim][:, i]; kw...)
end

function getfactor(t::TensorDecompositions.Tucker, dim::Integer=1, cuttoff::Number=0)
	i = vec(maximum(t.factors[dim], 1) .> cuttoff)
	s = size(t.factors[dim])
	println("Factor $dim: size $s -> ($(s[1]), $(sum(i)))")
	t.factors[dim][:, i]
end

function plotfactors(t::TensorDecompositions.Tucker, cuttoff::Number=0; kw...)
	for i = 1:length(t.factors)
		display(plotfactor(t, i, cuttoff; kw...))
		println()
	end
end

function plotcore(t::TensorDecompositions.Tucker, dim::Integer=1, cuttoff::Number=0; kw...)
	plottensor(t.core, dim; cutoff=true, cutvalue=cuttoff)
end

function plottensor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...)
	X = TensorDecompositions.compose(t)
	plottensor(X, dim; kw...)
end

function plottensor{T,N}(X::Array{T,N}, dim::Integer=1; minvalue=minimum(X), maxvalue=maximum(X), prefix::String="", movie::Bool=false, title="", hsize=6Compose.inch, vsize=6Compose.inch, moviedir::String=".", quiet::Bool=false, cleanup::Bool=true, sizes=size(X), timestep=1/sizes[dim], progressbar::Bool=true, mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), colormap=colormap_gyr, cutoff::Bool=false, cutvalue::Number=0)
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	if dim > N || dim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		M = X[nt...]
		if cutoff && maximum(M) .<= cutvalue
			continue
		end
		g = plotmatrix(M, minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap)
		if progressbar
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0.001Compose.h),
			(Compose.context(), Compose.fill("gray"), Compose.fontsize(10Compose.pt), Compose.text(0.01, -50000.0, sprintf("%6.4f", i * timestep), Compose.hleft, Compose.vtop)),
			(Compose.context(), Compose.fill("tomato"), Compose.rectangle(0.5, -50000.0, i/sizes[dim]*0.48, 15000.0)),
			(Compose.context(), Compose.fill("gray"), Compose.rectangle(0.5, -50000.0, 0.48, 15000.0)))
			p = Compose.vstack(g, f)
		else
			p = g
		end
		!quiet && (println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize), p); println())
		if prefix != ""
			filename = setnewfilename(prefix, i)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), p)
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
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
		cleanup && run(`find $moviedir -name $prefix-"frame*".png -delete`)
	end
end

function zerotensorcomponents(t::TensorDecompositions.Tucker, d::Int)
	t.core[nt...] .= 0
end

function zerotensorcomponents(t::TensorDecompositions.CANDECOMP, d::Int)
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

function plottensorcomponents(X1::Array, t2::TensorDecompositions.CANDECOMP; prefix::String="", filter=(), kw...)
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
		NTFk.plot2tensors(X1, X2; progressbar=false, prefix=prefix * string(i), kw...)
	end
end

function plottensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t2.core), prefix::String="", filter=(), kw...)
	ndimensons = length(size(X1))
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	pt = Vector{Int64}(0)
	if pdim > 1
		push!(pt, pdim)
		for i = ndimensons:-1:1
			if i != pdim
				push!(pt, i)
			end
		end
	else
		for i = 1:ndimensons
			push!(pt, i)
		end
	end
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
		NTFk.plot2tensors(permutedims(X1, pt), permutedims(X2, pt); progressbar=false, title=title, prefix=prefix * string(i),  kw...)
	end
end

function plot2tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t.core), prefix::String="", filter=(), order=[], kw...)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = Vector{Int64}(0)
	if pdim > 1
		push!(pt, pdim)
		for i = ndimensons:-1:1
			if i != pdim
				push!(pt, i)
			end
		end
	else
		for i = 1:ndimensons
			push!(pt, i)
		end
	end
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
		tt.core .= t.core
	end
	if sizeof(order) == 0
		order = gettensorcomponentorder(t, dim; method=:factormagnitudect)
	end
	NTFk.plot2tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt); prefix=prefix, kw...)
end

function plot2tensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t2.core), prefix::String="", filter=(), order=[], kw...)
	ndimensons = length(size(X1))
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = Vector{Int64}(0)
	if pdim > 1
		push!(pt, pdim)
		for i = ndimensons:-1:1
			if i != pdim
				push!(pt, i)
			end
		end
	else
		for i = 1:ndimensons
			push!(pt, i)
		end
	end
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
		tt.core .= t2.core
	end
	if sizeof(order) == 0
		order = gettensorcomponentorder(t2, dim; method=:factormagnitudect)
	end
	NTFk.plot3tensors(permutedims(X1, pt), permutedims(X2[order[1]], pt), permutedims(X2[order[2]], pt); prefix=prefix, kw...)
end

function plottensorandcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t.core), sizes=size(X), timestep=1/sizes[dim], cleanup::Bool=true, movie::Bool=false, moviedir=".", prefix::String="", title="", quiet::Bool=false, filter=(), order=[], minvalue=minimum(X), maxvalue=maximum(X), hsize=12Compose.inch, vsize=12Compose.inch, colormap=colormap_gyr, kw...)
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	ndimensons = length(sizes)
	if dim > ndimensons || dim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	if pdim > ndimensons || pdim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	dimname = namedimension(ndimensons; char="D", names=("Row", "Column", "Layer"))
	# s2 = plot2dmodtensorcomponents(X, t, dim, "mean"; xtitle="Time", ytitle="Mean concentrations", quiet=true, code=true)
	s2 = plot2dmodtensorcomponents(X, t, dim, "maximum"; xtitle="Time", ytitle="Max concentrations", quiet=true, code=true)
	# s2 = plot2dtensorcomponents(t, dim; xtitle="Time", ytitle="Component", quiet=true, code=true)
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : Colon()), ndimensons)
		p1 = plotmatrix(X[nt...], minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap)
		p2 = Gadfly.plot(s2..., Gadfly.layer(xintercept=[i*timestep], Gadfly.Geom.vline(color=["gray"], size=[2Gadfly.pt])))
		p = Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Gadfly.render(p1)),
							Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(p2)))
		!quiet && (println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize, dpi=50), p); println())
		if prefix != ""
			filename = setnewfilename(prefix, i)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), p)
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
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
		cleanup && run(`find $moviedir -name $prefix-"frame*".png -delete`)
	end

end

function plot3tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t.core), prefix::String="", filter=(), order=[], kw...)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 2
	pt = Vector{Int64}(0)
	if pdim > 1
		push!(pt, pdim)
		for i = ndimensons:-1:1
			if i != pdim
				push!(pt, i)
			end
		end
	else
		for i = 1:ndimensons
			push!(pt, i)
		end
	end
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
		tt.core .= t.core
	end
	if sizeof(order) == 0
		order = gettensorcomponentorder(t, dim; method=:factormagnitudect)
	end
	NTFk.plot3tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt), permutedims(X[order[3]], pt); prefix=prefix, kw...)
end

function plot2tensors(X1::Array, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...)
	X2 = TensorDecompositions.compose(T2)
	plotcmptensor(X1, X2, dim; kw...)
end

function plot2tensors{T,N}(X1::Array{T,N}, X2::Array{T,N}, dim::Integer=1; minvalue=minimum([X1 X2]), maxvalue=maximum([X1 X2]), prefix::String="", movie::Bool=false, hsize=12Compose.inch, vsize=6Compose.inch, title::String="", moviedir::String=".", ltitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timestep=1/sizes[dim], progressbar::Bool=true, mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), colormap=colormap_gyr)
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	@assert sizes == size(X2)
	if dim > N || dim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		g1 = plotmatrix(X1[nt...], minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=colormap)
		g2 = plotmatrix(X2[nt...], minvalue=minvalue, maxvalue=maxvalue, title=rtitle, colormap=colormap)
		g = Compose.hstack(g1, g2)
		if title != ""
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0.0001Compose.h),
							(Compose.context(), Compose.fill("gray"), Compose.fontsize(20Compose.pt), Compose.text(0.5Compose.w, 0, title * " : " * sprintf("%06d", i), Compose.hcenter, Compose.vtop)))
			g = Compose.vstack(t, g)
		end
		if progressbar
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0.001Compose.h),
				(Compose.context(), Compose.fill("gray"), Compose.fontsize(10Compose.pt), Compose.text(0.01, -50000.0, sprintf("%6.4f", i * timestep), Compose.hleft, Compose.vtop)),
				(Compose.context(), Compose.fill("tomato"), Compose.rectangle(0.5, -50000.0, i/sizes[dim]*0.48, 15000.0)),
				(Compose.context(), Compose.fill("gray"), Compose.rectangle(0.5, -50000.0, 0.48, 15000.0)))
			p = Compose.vstack(g, f)
		else
			p = g
		end
		!quiet && println(framename)
		!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize), p); println())
		if prefix != ""
			filename = setnewfilename(prefix, i)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), p)
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
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
		cleanup && run(`find $moviedir -name $prefix-"frame*.png" -delete`)
	end
end

plotcmptensor = plot2tensors

function plot3tensors{T,N}(X1::Array{T,N}, X2::Array{T,N}, X3::Array{T,N}, dim::Integer=1; minvalue=minimum([X1 X2 X3]), maxvalue=maximum([X1 X2 X3]), prefix::String="", movie::Bool=false, hsize=24Compose.inch, vsize=6Compose.inch, moviedir::String=".", ltitle::String="", ctitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timestep=1/sizes[dim], progressbar::Bool=true, mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), colormap=colormap_gyr, kw...)
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	@assert sizes == size(X2)
	@assert sizes == size(X3)
	if dim > N || dim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		g1 = plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=colormap, kw...)
		g2 = plotmatrix(X2[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ctitle, colormap=colormap, kw...)
		g3 = plotmatrix(X3[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=rtitle, colormap=colormap, kw...)
		g = Compose.hstack(g1, g2, g3)
		if progressbar
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0.001Compose.h),
				(Compose.context(), Compose.fill("gray"), Compose.fontsize(10Compose.pt), Compose.text(0.01, -25000.0, sprintf("%6.4f", i * timestep), Compose.hleft, Compose.vtop)),
				(Compose.context(), Compose.fill("tomato"), Compose.rectangle(0.75, -25000.0, i/sizes[dim]*0.2, 15000.0)),
				(Compose.context(), Compose.fill("gray"), Compose.rectangle(0.75, -25000.0, 0.2, 15000.0)))
			p = Compose.vstack(g, f)
		else
			p = g
		end
		!quiet && println(framename)
		!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize), p); println())
		if prefix != ""
			filename = setnewfilename(prefix, i)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), p)
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
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
		cleanup && run(`find $moviedir -name $prefix-"frame*.png" -delete`)
	end
end

function plotlefttensor(X1::Array, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...)
	X2 = TensorDecompositions.compose(T2)
	plot3tensors(X1, X2, X2-X1, dim; kw...)
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
	if !contains(fn, keyword)
		fn = root * "-$(keyword)000000." * ext
	end
	if ismatch(Regex(string("-", keyword, "[0-9]*\..*\$")), fn)
		rm = match(Regex(string("-", keyword, "([0-9]*)\.(.*)\$")), fn)
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
		warn("setnewfilename failed!")
		return ""
	end
end

function plot2d(T::Array, Te::Array; quiet::Bool=false, ymin=nothing, ymax=nothing, wellnames=nothing, Tmax=nothing, Tmin=nothing, xtitle::String="x", ytitle::String="y", figuredir="results", hsize=8Gadfly.inch, vsize=4Gadfly.inch, keyword="", dimname="Well", colors=[parse(Colors.Colorant, "green"), parse(Colors.Colorant, "orange"), parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "gray")], gm=[Gadfly.Guide.manual_color_key("", ["Oil", "Gas", "Water"], colors[1:3])])
	c = size(T)
	if wellnames != nothing
		@assert length(wellnames) == c[1]
	end
	@assert c == size(Te)
	if Tmax != nothing && Tmin != nothing
		@assert size(Tmax) == size(Tmin)
		append = ""
	else
		append = "_normalized"
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
		p = Vector{Any}(c[3] * 2)
		pc = 1
		for i = 1:c[3]
			y = T[w,:,i]
			ye = Te[w,:,i]
			if Tmax != nothing && Tmin != nothing
				y = y * (Tmax[w,i] - Tmin[w,i]) + Tmin[w,i]
				ye = ye * (Tmax[w,i] - Tmin[w,i]) + Tmin[w,i]
			end
			p[pc] = Gadfly.layer(x=1:c[2], y=y, Gadfly.Geom.line, Gadfly.Theme(line_width=1Gadfly.pt, default_color=colors[i]))
			pc += 1
			p[pc] = Gadfly.layer(x=1:c[2], y=ye, Gadfly.Geom.line, Gadfly.Theme(line_width=1Gadfly.pt, line_style=:dash, default_color=colors[i]))
			pc += 1
		end
		tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
		if wellnames != nothing
			tm = [tm..., Gadfly.Guide.title("$dimname $(wellnames[w])")]
			filename = "$(figuredir)/$(lowercase(dimname))_$(wellnames[w])$(append).png"
		else
			filename = "$(figuredir)/$(lowercase(dimname))$(append).png"
		end
		f = Gadfly.plot(p..., tm..., Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...)
		Gadfly.draw(Gadfly.PNG(filename, hsize, vsize, dpi=300), f)
		!quiet && (display(f); println())
	end
end

"Convert `@sprintf` macro into `sprintf` function"
sprintf(args...) = eval(:@sprintf($(args...)))
