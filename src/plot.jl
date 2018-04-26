import Gadfly
import Colors
import Compose
import TensorToolbox
import TensorDecompositions
import Distributions
import Interpolations

colors = ["red", "blue", "green", "orange", "magenta", "cyan", "brown", "pink", "lime", "navy", "maroon", "yellow", "olive", "springgreen", "teal", "coral", "lavender", "beige"]
ncolors = length(colors)

# r = reshape(repeat(collect(1/100:1/100:1), inner=100), (100,100))
# NTFk.plotmatrix(r; colormap=NTFk.colormap_hsv2);

colormap_hsv2 = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "brown4"), parse(Colors.Colorant, "coral"), parse(Colors.Colorant, "darkmagenta"), parse(Colors.Colorant, "peachpuff"), parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "cyan"), parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"))]
colormap_hsv = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "magenta"), parse(Colors.Colorant, "peachpuff"), parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "cyan"), parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"))]
colormap_rbw2 = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "cyan"), parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"), parse(Colors.Colorant, "darkmagenta"))]
colormap_rbw = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "cyan"), parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"))]
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
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
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

function getgridvalues(v, r; logtransform=true)
	lv = length(v)
	lr = length(r)
	@assert lv == lr
	f = similar(v)
	for i=1:lv
		try
			if logtransform
				f[i] = Interpolations.interpolate((log10.(r[i]),), 1:length(r[i]), Interpolations.Gridded(Interpolations.Linear()))[log10(v[i])]
			else
				f[i] = Interpolations.interpolate((r[i],), 1:length(r[i]), Interpolations.Gridded(Interpolations.Linear()))[v[i]]
			end
		catch
			if logtransform
				f[i] = Interpolations.interpolate((sort!(log10.(r[i])),), length(r[i]):-1:1, Interpolations.Gridded(Interpolations.Linear()))[log10(v[i])]
			else
				f[i] = Interpolations.interpolate((sort!(r[i]),), length(r[i]):-1:1, Interpolations.Gridded(Interpolations.Linear()))[v[i]]
			end
		end
	end
	return f
end

function getinterpolatedtensor{T,N}(t::TensorDecompositions.Tucker{T,N}, v; sp=[Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line())), Interpolations.OnCell()])
	lv = length(v)
	f = Vector(lv)
	factors = []
	for i = 1:N
		push!(factors, t.factors[i])
	end
	for j = 1:lv
		if !isnan(v[j])
			cv = size(t.factors[j], 2)
			f = Array{T}(1,cv)
			for i = 1:cv
				f[1, i] = Interpolations.interpolate(t.factors[j][:, i], sp...)[v[j]]
			end
			factors[j] = f
		end
	end
	tn = TensorDecompositions.Tucker((factors...), t.core)
	return tn
end

function gettensorcomponentorder(t::TensorDecompositions.Tucker, dim::Integer=1; method::Symbol=:core, firstpeak::Bool=true, quiet=true)
	cs = size(t.core)[dim]
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	if method == :factormagnitude
		fmax = vec(maximum(t.factors[dim], 1)) .- vec(minimum(t.factors[dim], 1))
		@assert cs == length(fmax)
		for i = 1:cs
			if fmax[i] == 0
				warn("Maximum of component $i is equal to zero!")
			end
		end
		ifmax = sortperm(fmax; rev=true)[1:crank]
		!quiet && info("Max factor magnitudes: $fmax")
		if firstpeak
			imax = map(i->indmax(t.factors[dim][:, ifmax[i]]), 1:crank)
			order = ifmax[sortperm(imax)]
		else
			order = ifmax[1:crank]
		end
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
		!quiet && info("Max core magnitudes: $maxXe")
		imax = sortperm(maxXe; rev=true)
		order = imax[1:crank]
	end
	return order
end

function gettensorminmax(t::TensorDecompositions.Tucker, dim::Integer=1; method::Symbol=:core)
	cs = size(t.core)[dim]
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	if method == :factormagnitude
		fmin = vec(minimum(t.factors[dim], 1))
		fmax = vec(maximum(t.factors[dim], 1))
		@assert cs == length(fmax)
		for i = 1:cs
			if fmax[i] == 0
				warn("Maximum of component $i is equal to zero!")
			end
		end
		info("Max factor magnitudes: $fmax")
		info("Min factor magnitudes: $fmin")
	elseif method == :all
		Te = TensorDecompositions.compose(t)
		tsize = size(Te)
		ts = tsize[dim]
		maxTe = Vector{Float64}(ts)
		minTe = Vector{Float64}(ts)
		for i = 1:tsize[dim]
			nt = ntuple(k->(k == dim ? i : Colon()), ndimensons)
			minTe[i] = minimum(Te[nt...])
			maxTe[i] = maximum(Te[nt...])
		end
		info("Max all magnitudes: $maxTe")
		info("Min all magnitudes: $minTe")
	else
		maxXe = Vector{Float64}(cs)
		minXe = Vector{Float64}(cs)
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
		info("Max core magnitudes: $maxXe")
		info("Min core magnitudes: $minXe")
	end
end

function gettensorcomponentgroups(t::TensorDecompositions.Tucker, dim::Integer=1; cutvalue::Number=0.9)
	g = zeros(t.factors[dim][:, 1])
	v = maximum(t.factors[dim], 1) .> cutvalue
	gi = 0
	for i = 1:length(v)
		if v[i]
			m = t.factors[dim][:, i] .> cutvalue
			gi += 1
			g[m] = gi
		end
	end
	info("Number of component groups in dimension $dim is $(gi)")
	return g
end

function plot2dtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude))
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	componentnames = map(i->"T$i", 1:crank)
	p = t.factors[dim]
	pl = Vector{Any}(crank)
	for i = 1:crank
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=p[:, order[i]], Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tx = timescale ? [] : [Gadfly.Coord.Cartesian(xmin=minimum(xvalues), xmax=maximum(xvalues))]
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:crank])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., tm...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., tm..., tx...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, functionname::String="mean"; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude))
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
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
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tx = timescale ? [] : [Gadfly.Coord.Cartesian(xmin=minimum(xvalues), xmax=maximum(xvalues))]
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:crank])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., tc...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., tc..., tx...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, functionname1::String="mean", functionname2::String="mean"; quiet=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude))
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
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

function plotmatrix(X::Matrix; minvalue=minimumnan(X), maxvalue=maximumnan(X), label="", title="", xlabel="", ylabel="", gm=[Gadfly.Guide.xticks(label=false, ticks=nothing), Gadfly.Guide.yticks(label=false, ticks=nothing)], masize::Int64=0, colormap=colormap_gyr, filename::String="", hsize=6Compose.inch, vsize=6Compose.inch, figuredir::String=".")
	Xp = min.(max.(movingaverage(X, masize), minvalue), maxvalue)
	p = Gadfly.spy(Xp, Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Guide.ColorKey(title=label), Gadfly.Scale.ContinuousColorScale(colormap..., minvalue=minvalue, maxvalue=maxvalue), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt, bar_spacing=0Gadfly.mm), gm...)
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=300), p)
	end
	return p
end

function plotfactor(t::TensorDecompositions.Tucker, dim::Integer=1, cuttoff::Number=0; kw...)
	plotmatrix(getfactor(t, dim, cuttoff); kw...)
end

function getfactor(t::TensorDecompositions.Tucker, dim::Integer=1, cuttoff::Number=0)
	i = vec(maximum(t.factors[dim], 1) .> cuttoff)
	s = size(t.factors[dim])
	println("Factor $dim: size $s -> ($(s[1]), $(sum(i)))")
	t.factors[dim][:, i]
end

function plotfactors(t::TensorDecompositions.Tucker, cuttoff::Number=0; prefix="", kw...)
	for i = 1:length(t.factors)
		display(plotfactor(t, i, cuttoff; filename="$(prefix)_factor$(i).png", kw...))
		println()
	end
end

function plotcore(t::TensorDecompositions.Tucker, dim::Integer=1, cuttoff::Number=0; kw...)
	plottensor(t.core, dim; progressbar=nothing, cutoff=true, cutvalue=cuttoff, kw...)
end

function plottensor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; mask=nothing, offset=0, kw...)
	X = TensorDecompositions.compose(t)
	if mask != nothing
		X[mask] = NaN
	end
	if offset != 0
		X .+ offset
	end
	plottensor(X, dim; kw...)
end

function plottensor{T,N}(X::Array{T,N}, dim::Integer=1; minvalue=minimumnan(X), maxvalue=maximumnan(X), prefix::String="", keyword="frame", movie::Bool=false, title="", hsize=6Compose.inch, vsize=6Compose.inch, moviedir::String=".", quiet::Bool=false, cleanup::Bool=true, sizes=size(X), timescale::Bool=true, timestep=1/sizes[dim], progressbar=progressbar_regular, mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), colormap=colormap_gyr, cutoff::Bool=false, cutvalue::Number=0)
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
		if cutoff && maximumnan(M) .<= cutvalue
			continue
		end
		g = plotmatrix(M, minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap)
		if progressbar != nothing
			f = progressbar(i, timescale, timestep)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		!quiet && (println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(g, f)); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(g, f))
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
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
		plot2tensors(X1, X2; progressbar=nothing, prefix=prefix * string(i), kw...)
	end
end

function plottensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=true, csize::Tuple=TensorToolbox.mrank(t2.core), prefix::String="", filter=(), kw...)
	ndimensons = length(size(X1))
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	pt = Vector{Int64}(0)
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
		plot2tensors(permutedims(X1, pt), permutedims(X2, pt); progressbar=nothing, title=title, prefix=prefix * string(i),  kw...)
	end
end

function plot2tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=true,csize::Tuple=TensorToolbox.mrank(t.core), mask=nothing, offset=0, prefix::String="", filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude), kw...)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = Vector{Int64}(0)
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
		if mask != nothing
			X[i][mask] = NaN
		end
		if offset != 0
			X[i] .+ offset
		end
		tt.core .= t.core
	end
	plot2tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt); prefix=prefix, kw...)
end

function plot2tensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=true, csize::Tuple=TensorToolbox.mrank(t2.core), mask=nothing, offset=0, prefix::String="", filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude), kw...)
	ndimensons = length(size(X1))
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = Vector{Int64}(0)
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
		if mask != nothing
			X2[i][mask] = NaN
		end
		if offset != 0
			X2[i] .+ offset
		end
		tt.core .= t2.core
	end
	plot3tensors(permutedims(X1, pt), permutedims(X2[order[1]], pt), permutedims(X2[order[2]], pt); prefix=prefix, kw...)
end

function plottensorandcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t.core), sizes=size(X), xtitle="Time", ytitle="Magnitude", timescale::Bool=true, timestep=1/sizes[dim], cleanup::Bool=true, movie::Bool=false, moviedir=".", prefix::String="", keyword="frame", title="", quiet::Bool=false, filter=(), minvalue=minimumnan(X), maxvalue=maximumnan(X), hsize=12Compose.inch, vsize=12Compose.inch, colormap=colormap_gyr, kw...)
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
	s2 = plot2dmodtensorcomponents(X, t, dim, "maximum"; xtitle=xtitle, ytitle=ytitle, timescale=timescale, quiet=true, code=true)
	progressbar_2d = make_progressbar_2d(s2)
	# s2 = plot2dtensorcomponents(t, dim; xtitle="Time", ytitle="Component", quiet=true, code=true)
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : Colon()), ndimensons)
		p1 = plotmatrix(X[nt...], minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap)
		p2 = progressbar_2d(i, timescale, timestep)
		!quiet && (println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Gadfly.render(p1)), Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(p2)))); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Gadfly.render(p1)), Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(p2))))
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
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

function plot3tensorsandcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; xtitle="Time", ytitle="Max concentrations", timescale::Bool=true, kw...)
	ndimensons = length(t.factors)
	if dim > ndimensons || dim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	if pdim > ndimensons || pdim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	timestep = 1 / size(t.factors[dim], 1)
	s2 = plot2dtensorcomponents(t, dim; xtitle=xtitle, ytitle=ytitle, timescale=timescale, quiet=true, code=true)
	progressbar_2d = make_progressbar_2d(s2)
	plot3tensorcomponents(t, dim; timescale=timescale, quiet=false, progressbar=progressbar_2d, hsize=12Compose.inch, vsize=6Compose.inch, kw...)
end

function plot3tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=true,csize::Tuple=TensorToolbox.mrank(t.core), prefix::String="", filter=(), mask=nothing, offset=0, order=gettensorcomponentorder(t, dim; method=:factormagnitude), kw...)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 2
	pt = Vector{Int64}(0)
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
		if offset != 0
			X[i] .+ offset
		end
		if mask != nothing
			X[i][mask] = NaN
		end
		tt.core .= t.core
	end
	plot3tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt), permutedims(X[order[3]], pt); prefix=prefix, kw...)
end

function plot2tensors(X1::Array, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...)
	X2 = TensorDecompositions.compose(T2)
	plot2tensors(X1, X2, dim; kw...)
end

function plot2tensors{T,N}(X1::Array{T,N}, X2::Array{T,N}, dim::Integer=1; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), minvalue2=minvalue, maxvalue2=maxvalue, movie::Bool=false, hsize=12Compose.inch, vsize=6Compose.inch, title::String="", moviedir::String=".", prefix::String = "", keyword="frame", ltitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timescale::Bool=true, timestep=1/sizes[dim], progressbar=progressbar_regular, mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), uniformscaling::Bool=true, colormap=colormap_gyr)
	if !uniformscaling
		minvalue=minimumnan(X1)
		maxvalue=maximumnan(X1)
		minvalue2=minimumnan(X2)
		maxvalue2=maximumnan(X2)
	end
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
		g2 = plotmatrix(X2[nt...], minvalue=minvalue2, maxvalue=maxvalue2, title=rtitle, colormap=colormap)
		if title != ""
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0.0001Compose.h),
							(Compose.context(), Compose.fill("gray"), Compose.fontsize(20Compose.pt), Compose.text(0.5Compose.w, 0, title * " : " * sprintf("%06d", i), Compose.hcenter, Compose.vtop)))
		else
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		if progressbar != nothing
			f = progressbar(i, timescale, timestep)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		!quiet && println(framename)
		!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(t, Compose.hstack(g1, g2), f)); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(t, Compose.hstack(g1, g2), f))
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
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

plotcmptensors = plot2tensors

function plot3tensors{T,N}(X1::Array{T,N}, X2::Array{T,N}, X3::Array{T,N}, dim::Integer=1; minvalue=minimumnan([X1 X2 X3]), maxvalue=maximumnan([X1 X2 X3]), minvalue2=minvalue, maxvalue2=maxvalue, minvalue3=minvalue, maxvalue3=maxvalue, prefix::String="", keyword="frame", movie::Bool=false, hsize=24Compose.inch, vsize=6Compose.inch, moviedir::String=".", ltitle::String="", ctitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timescale::Bool=true, timestep=1/sizes[dim], progressbar=progressbar_regular, largebar::Bool=true, mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), colormap=colormap_gyr, uniformscaling::Bool=true, kw...)
	if !uniformscaling
		minvalue=minimumnan(X1)
		maxvalue=maximumnan(X1)
		minvalue2=minimumnan(X2)
		maxvalue2=maximumnan(X2)
		minvalue3=minimumnan(X3)
		maxvalue3=maximumnan(X3)
	end
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
		g2 = plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, title=ctitle, colormap=colormap, kw...)
		g3 = plotmatrix(X3[nt...]; minvalue=minvalue3, maxvalue=maxvalue3, title=rtitle, colormap=colormap, kw...)
		if progressbar != nothing
			f = progressbar(i, timescale, timestep)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		if !quiet
			println(framename)
			if largebar
				Gadfly.draw(Gadfly.PNG(hsize, vsize), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Compose.hstack(g1, g2, g3)), Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(f)))); println()
			else
				Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(Compose.hstack(g1, g2, g3), f)); println()
			end
		end
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			if largebar
				Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Compose.hstack(g1, g2, g3)), Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(f))))
			else
				Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(Compose.hstack(g1, g2, g3), f))
			end
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
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

function plotlefttensor(X1::Array, X2::Array, dim::Integer=1; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), center=true, kw...)
	D=X2-X1
	min3 = minimumnan(D)
	max3 = maximumnan(D)
	if center
		min3, max3 = min(min3, -max3), max(max3, -min3)
	end
	plot3tensors(X1, X2, D, dim; minvalue=minvalue, maxvalue=maxvalue, minvalue3=min3, maxvalue3=max3, kw...)
end

function plotlefttensor(X1::Array, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; center=true, kw...)
	X2 = TensorDecompositions.compose(T2)
	D=X2-X1
	min3 = minimumnan(D)
	max3 = maximumnan(D)
	if center
		min3, max3 = min(min3, -max3), max(max3, -min3)
	end
	plot3tensors(X1, X2, D, dim; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), minvalue3=min3, maxvalue3=max3, kw...)
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

function plot2d(T::Array, Te::Array; quiet::Bool=false, ymin=nothing, ymax=nothing, wellnames=nothing, Tmax=nothing, Tmin=nothing, xtitle::String="x", ytitle::String="y", figuredir="results", hsize=8Gadfly.inch, vsize=4Gadfly.inch, keyword="", dimname="Well", colors=[parse(Colors.Colorant, "green"), parse(Colors.Colorant, "orange"), parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "gray")], gm=[Gadfly.Guide.manual_color_key("", ["Oil", "Gas", "Water"], colors[1:3]), Gadfly.Theme(major_label_font_size=16Gadfly.pt, key_label_font_size=14Gadfly.pt, minor_label_font_size=12Gadfly.pt)])
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
			p[pc] = Gadfly.layer(x=1:c[2], y=y, Gadfly.Geom.line, Gadfly.Theme(line_width=2Gadfly.pt, default_color=colors[i]))
			pc += 1
			p[pc] = Gadfly.layer(x=1:c[2], y=ye, Gadfly.Geom.line, Gadfly.Theme(line_width=2Gadfly.pt, line_style=:dash, default_color=colors[i]))
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

function maximumnan(X; kw...)
	maximum(X[.!isnan.(X)]; kw...)
end

function minimumnan(X; kw...)
	minimum(X[.!isnan.(X)]; kw...)
end

function progressbar_regular(i::Number, timescale::Bool=false, timestep::Number=1)
	s = timescale ? sprintf("%6.4f", i * timestep) : sprintf("%6d", i)
	return Compose.compose(Compose.context(0, 0, 1Compose.w, 0.05Compose.h),
		(Compose.context(), Compose.fill("gray"), Compose.fontsize(10Compose.pt), Compose.text(0.01, 0.0, s, Compose.hleft, Compose.vtop)),
		(Compose.context(), Compose.fill("tomato"), Compose.rectangle(0.75, 0.0, i * timestep * 0.2, 5)),
		(Compose.context(), Compose.fill("gray"), Compose.rectangle(0.75, 0.0, 0.2, 5)))
end

function make_progressbar_2d(s)
	function progressbar_2d(i::Number, timescale::Bool=false, timestep::Number=1)
		xi = timescale ? i * timestep : i
		return Gadfly.plot(s..., Gadfly.layer(xintercept=[xi], Gadfly.Geom.vline(color=["gray"], size=[2Gadfly.pt])))
	end
	return progressbar_2d
end