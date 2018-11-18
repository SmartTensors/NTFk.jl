import Gadfly
import Measures
import Colors
import Compose
import TensorToolbox
import TensorDecompositions
import Distributions

function plotmatrix(X::AbstractVector; kw...)
	plotmatrix(convert(Array{Float64,2}, permutedims(X)); kw...)
end

function plotmatrix(X::AbstractMatrix; minvalue=minimumnan(X), maxvalue=maximumnan(X), label="", title="", xlabel="", ylabel="", xticks=nothing, yticks=nothing, gm=[Gadfly.Guide.xticks(label=false, ticks=nothing), Gadfly.Guide.yticks(label=false, ticks=nothing)], masize::Int64=0, colormap=colormap_gyr, filename::String="", hsize=6Compose.inch, vsize=6Compose.inch, figuredir::String=".", colorkey::Bool=true, mask=nothing, polygon=nothing, contour=nothing, linewidth::Measures.Length{:mm,Float64}=1Gadfly.pt, linecolor="gray", defaultcolor=nothing, pointsize=1Gadfly.pt, transform=nothing, code=false)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	Xp = deepcopy(min.(max.(movingaverage(X, masize), minvalue), maxvalue))
	if transform != nothing
		Xp = transform.(Xp)
	end
	nanmask!(Xp, mask)
	if xticks != nothing
		gm = [gm..., Gadfly.Scale.x_discrete(labels=i->xticks[i]), Gadfly.Guide.xticks(label=true)]
	end
	if yticks != nothing
		gm = [gm..., Gadfly.Scale.y_discrete(labels=i->yticks[i]), Gadfly.Guide.yticks(label=true)]
	end
	cs = colorkey ? [Gadfly.Guide.ColorKey(title=label)] : [Gadfly.Theme(key_position = :none)]
	cm = colormap == nothing ? [] : [Gadfly.Scale.ContinuousColorScale(colormap..., minvalue=minvalue, maxvalue=maxvalue)]
	cs = colormap == nothing ? [] : cs
	ds = min.(size(Xp)) == 1 ? [Gadfly.Scale.x_discrete, Gadfly.Scale.y_discrete] : []
	is, js, xs = Gadfly._findnz(x->!isnan(x), Xp)
	n, m = size(Xp)
	gt = [Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt, bar_spacing=0Gadfly.mm), Gadfly.Scale.x_continuous, Gadfly.Scale.y_continuous, Gadfly.Coord.cartesian(yflip=true, fixed=true, xmin=0.5, xmax=m+.5, ymin=0.5, ymax=n+.5)]
	l = defaultcolor == nothing ? [Gadfly.layer(x=js, y=is, color=xs, Gadfly.Geom.rectbin())] : [Gadfly.layer(x=js, y=is, Gadfly.Theme(default_color=defaultcolor, point_size=pointsize, highlight_width=0Gadfly.pt))]
	if polygon == nothing && contour == nothing
		c = l..., ds..., cm..., cs..., gm..., gt...
	else
		if polygon != nothing
			c = Gadfly.layer(x=polygon[1], y=polygon[2], Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=linecolor)), l..., ds..., cm..., cs..., gm..., gt...
		else
			c = Gadfly.layer(z=permutedims(contour .* (maxvalue - minvalue) .+ minvalue), x=collect(1:size(contour, 2)), y=collect(1:size(contour, 1)), Gadfly.Geom.contour(levels=[minvalue]), Gadfly.Theme(line_width=linewidth, default_color=linecolor)), l..., ds..., cm..., cs..., gm..., gt...
		end
	end
	p = Gadfly.plot(c...)
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=300), p)
	end
	if code
		return c
	else
		return p
	end
end

function plotfactor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1, cutoff::Number=0; kw...)
	plotmatrix(getfactor(t, dim, cutoff); kw...)
end

function plotfactors(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, cutoff::Number=0; prefix="", kw...)
	recursivemkdir(prefix)
	for i = 1:length(t.factors)
		display(plotfactor(t, i, cutoff; filename="$(prefix)_factor$(i).png", kw...))
		println()
	end
end

function plotcore(t::TensorDecompositions.Tucker, dim::Integer=1, cutoff::Number=0; kw...)
	plottensor(t.core, dim; progressbar=nothing, cutoff=true, cutvalue=cutoff, kw...)
end

function plottensor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; mask=nothing, transform=nothing, kw...)
	X = TensorDecompositions.compose(t)
	if transform != nothing
		X = transform.(X)
	end
	if typeof(mask) <: Number
		nanmask!(X, mask)
	else
		nanmask!(X, mask, dim)
	end
	plottensor(X, dim; kw...)
end

function plottensor(X::AbstractArray{T,N}, dim::Integer=1; mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=minimumnan(X), maxvalue=maximumnan(X), prefix::String="", keyword="frame", movie::Bool=false, title="", hsize=6Compose.inch, vsize=6Compose.inch, moviedir::String=".", quiet::Bool=false, cleanup::Bool=true, sizes=size(X), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=(datestart != nothing) ? datestart + eval(parse(dateincrement))(sizes[dim]) : nothing, dateincrement::String="Dates.Day", progressbar=progressbar_regular, colormap=colormap_gyr, cutoff::Bool=false, cutvalue::Number=0, vspeed=1.0, kw...) where {T,N}
	if !checkdimension(dim, N)
		return
	end
	recursivemkdir(moviedir; filename=false)
	recursivemkdir(prefix)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		M = X[nt...]
		if cutoff && maximumnan(M) .<= cutvalue
			continue
		end
		g = plotmatrix(M; minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap, kw...)
		if progressbar != nothing
			f = progressbar(i, timescale, timestep, datestart, dateend, dateincrement)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		!quiet && ((sizes[dim] > 1) && println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(g, f)); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(g, f))
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

function plot2matrices(X1::Matrix, X2::Matrix; kw...)
	plot2tensors([X1], [X2], 1; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), kw...)
end

function plot2tensors(X1::Array{T,N}, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...) where {T,N}
	X2 = convert(Array{T,N}, TensorDecompositions.compose(T2))
	plot2tensors(X1, X2, dim; kw...)
end

function plot2tensors(X1::AbstractArray{T,N}, X2::AbstractArray{T,N}, dim::Integer=1; mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), minvalue2=minvalue, maxvalue2=maxvalue, movie::Bool=false, hsize=12Compose.inch, vsize=6Compose.inch, title::String="", moviedir::String=".", prefix::String = "", keyword="frame", ltitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=(datestart != nothing) ? datestart + eval(parse(dateincrement))(sizes[dim]) : nothing, dateincrement::String="Dates.Day", progressbar=progressbar_regular, uniformscaling::Bool=true, colormap=colormap_gyr, vspeed=1.0, kw...) where {T,N}
	if !checkdimension(dim, N)
		return
	end
	recursivemkdir(prefix)
	if !uniformscaling
		minvalue = minimumnan(X1)
		maxvalue = maximumnan(X1)
		minvalue2 = minimumnan(X2)
		maxvalue2 = maximumnan(X2)
	end
	recursivemkdir(moviedir; filename=false)
	@assert sizes == size(X2)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		g1 = plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=colormap, kw...)
		g2 = plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, title=rtitle, colormap=colormap, kw...)
		if title != ""
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0.0001Compose.h),
				(Compose.context(), Compose.fill("gray"), Compose.fontsize(20Compose.pt), Compose.text(0.5Compose.w, 0, title * " : " * sprintf("%06d", i), Compose.hcenter, Compose.vtop)))
		else
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		if progressbar != nothing
			f = progressbar(i, timescale, timestep, datestart, dateend, dateincrement)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		!quiet && (sizes[dim] > 1) && println(framename)
		!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(t, Compose.hstack(g1, g2), f)); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(t, Compose.hstack(g1, g2), f))
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
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*.png" -delete`)
	end
end

function plot3matrices(X1::Matrix, X2::Matrix, X3::Matrix; kw...)
	plot3tensors([X1], [X2], [X3], 1; minvalue=minimumnan([X1 X2 X3]), maxvalue=maximumnan([X1 X2 X3]), kw...)
end

function plotcmptensors(X1::Array{T,N}, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; center=true, transform=nothing, mask=nothing, kw...) where {T,N}
	X2 = TensorDecompositions.compose(T2)
	if transform != nothing
		X2 = transform.(X2)
	end
	nanmask!(X2, mask)
	plot2tensors(X1, convert(Array{T,N}, X2), dim; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), kw...)
end

function plot3tensors(X1::AbstractArray{T,N}, X2::AbstractArray{T,N}, X3::AbstractArray{T,N}, dim::Integer=1; mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=minimumnan([X1 X2 X3]), maxvalue=maximumnan([X1 X2 X3]), minvalue2=minvalue, maxvalue2=maxvalue, minvalue3=minvalue, maxvalue3=maxvalue, prefix::String="", keyword="frame", movie::Bool=false, hsize=24Compose.inch, vsize=6Compose.inch, moviedir::String=".", ltitle::String="", ctitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", progressbar=progressbar_regular, barratio::Number=1/2, uniformscaling::Bool=true, vspeed=1.0, colormap=colormap_gyr, overlap::Bool=false, opacity=0.8, kw...) where {T,N}
	recursivemkdir(prefix)
	if !checkdimension(dim, N)
		return
	end
	if !uniformscaling
		minvalue = minimumnan(X1)
		maxvalue = maximumnan(X1)
		minvalue2 = minimumnan(X2)
		maxvalue2 = maximumnan(X2)
		minvalue3 = minimumnan(X3)
		maxvalue3 = maximumnan(X3)
	end
	recursivemkdir(moviedir; filename=false)
	@assert sizes == size(X2)
	@assert sizes == size(X3)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	if length(colormap) > 2
		colormap1 = colormap[1]; colormap2 = colormap[2]; colormap3 = colormap[3]
	else
		colormap1 = colormap2 = colormap3 = colormap
	end
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i / $(sizes[dim])"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		if overlap
			g1 = plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=nothing, defaultcolor=Colors.RGBA(1.0,0.0,0.0,opacity), code=overlap, kw...)
			g2 = plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, title=ctitle, colormap=nothing, defaultcolor=Colors.RGBA(0.0,0.0,1.0,opacity), code=overlap, kw...)
			g3 = plotmatrix(X3[nt...]; minvalue=minvalue3, maxvalue=maxvalue3, title=rtitle, colormap=nothing, defaultcolor=Colors.RGBA(0.0,0.502,0.0,opacity), code=overlap, kw...)
		else
			g1 = plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=colormap1, kw...)
			g2 = plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, title=ctitle, colormap=colormap2, kw...)
			g3 = plotmatrix(X3[nt...]; minvalue=minvalue3, maxvalue=maxvalue3, title=rtitle, colormap=colormap3, kw...)
		end
		g = overlap ? Compose.hstack(Gadfly.plot(g1..., g2..., g3...)) : Compose.hstack(g1, g2, g3)
		if progressbar != nothing
			if sizes[dim] == 1
				f = progressbar(0, timescale, timestep, datestart, dateend, dateincrement)
			else
				f = progressbar(i, timescale, timestep, datestart, dateend, dateincrement)
			end
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		if !quiet
			(sizes[dim] > 1) && println(framename)
			if typeof(f) != Compose.Context
				if overlap
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), g), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f)))); println()
				else
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), g), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f)))); println()
				end
			else
				if overlap
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(g, f)); println()
				else
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(g, f)); println()
				end
			end
		end
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			if typeof(f) != Compose.Context
				Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), g), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f))))
			else
				Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(g, f))
			end
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
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*.png" -delete`)
	end
end

function plotleftmatrix(X1::Matrix, X2::Matrix; kw...)
	plot3tensors([X1], [X2], [X2.-X1], 1; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), kw...)
end

function plotlefttensor(X1::Array, X2::Array, dim::Integer=1; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), minvalue3=nothing, maxvalue3=nothing, center=true, kw...)
	D = X2 - X1
	minvalue3 = minvalue3 == nothing ? minimumnan(D) : minvalue3
	maxvalue3 = maxvalue3 == nothing ? maximumnan(D) : maxvalue3
	if center
		minvalue3, maxvalue3 = min(minvalue3, -maxvalue3), max(maxvalue3, -minvalue3)
	end
	plot3tensors(X1, X2, D, dim; minvalue=minvalue, maxvalue=maxvalue, minvalue3=minvalue3, maxvalue3=maxvalue3, kw...)
end

function plotlefttensor(X1::Array{T,N}, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; minvalue=nothing, maxvalue=nothing, minvalue3=nothing, maxvalue3=nothing, center=true, transform=nothing, mask=nothing, kw...) where {T,N}
	X2 = TensorDecompositions.compose(T2)
	if transform != nothing
		X2 = transform.(X2)
	end
	D = X2 - X1
	nanmask!(X2, mask)
	nanmask!(D, mask)
	minvalue = minvalue == nothing ? minimumnan([X1 X2]) : minvalue
	maxvalue = maxvalue == nothing ? maximumnan([X1 X2]) : maxvalue
	minvalue3 = minvalue3 == nothing ? minimumnan(D) : minvalue3
	maxvalue3 = maxvalue3 == nothing ? maximumnan(D) : maxvalue3
	if center
		minvalue3, maxvalue3 = min(minvalue3, -maxvalue3), max(maxvalue3, -minvalue3)
	end
	plot3tensors(X1, convert(Array{T,N}, X2), convert(Array{T,N}, D), dim; minvalue=minvalue, maxvalue=maxvalue, minvalue3=minvalue3, maxvalue3=maxvalue3, kw...)
end

"""
colors=[parse(Colors.Colorant, "green"), parse(Colors.Colorant, "orange"), parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "gray")]
gm=[Gadfly.Guide.manual_color_key("", ["Oil", "Gas", "Water"], colors[1:3]), Gadfly.Theme(major_label_font_size=16Gadfly.pt, key_label_font_size=14Gadfly.pt, minor_label_font_size=12Gadfly.pt)]
"""
function plot2d(T::Array, Te::Array=T; quiet::Bool=false, wellnames=nothing, Tmax=nothing, Tmin=nothing, xtitle::String="", ytitle::String="", titletext::String="", figuredir::String="results", hsize=8Gadfly.inch, vsize=4Gadfly.inch, keyword::String="", dimname::String="Column", colors=NTFk.colors, gm=[Gadfly.Theme(major_label_font_size=16Gadfly.pt, key_label_font_size=14Gadfly.pt, minor_label_font_size=12Gadfly.pt)], linewidth::Measures.Length{:mm,Float64}=2Gadfly.pt, xaxis=1:size(Te,2), xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, xintercept=[])
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
		p = Vector{Any}(nlayers * 2)
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
			p[pc] = Gadfly.layer(x=xaxis, y=ye, xintercept=xintercept, Gadfly.Geom.line, Gadfly.Theme(line_style=:dot, line_width=linewidth, default_color=colors[i]), Gadfly.Geom.vline)
			pc += 1
		end
		if wellnames != nothing
			tm = [Gadfly.Guide.title("$dimname $(wellnames[w]) $titletext")]
			if dimname != ""
				filename = "$(figuredir)/$(lowercase(dimname))_$(wellnames[w])$(append).png"
			else
				filename = "$(figuredir)/$(wellnames[w])$(append).png"
			end
		else
			tm = []
			if dimname != ""
				filename = "$(figuredir)/$(lowercase(dimname))$(append).png"
			else
				filename = "$(figuredir)/$(append[2:end]).png"
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
		f = Gadfly.plot(p..., tm..., Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=yming, ymax=ymaxg))
		Gadfly.draw(Gadfly.PNG(filename, hsize, vsize, dpi=300), f)
		!quiet && (display(f); println())
	end
end

function progressbar_regular(i::Number, timescale::Bool=false, timestep::Number=1, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day")
	s = timescale ? sprintf("%6.4f", i * timestep) : sprintf("%6d", i)
	if datestart != nothing
		if dateend != nothing
			s = datestart + ((dateend .- datestart) * (i-1) * timestep)
		else
			s = datestart + eval(parse(dateincrement))(i-1)
		end
	end
	return Compose.compose(Compose.context(0, 0, 1Compose.w, 0.05Compose.h),
		(Compose.context(), Compose.fill("gray"), Compose.fontsize(10Compose.pt), Compose.text(0.01, 0.0, s, Compose.hleft, Compose.vtop)),
		(Compose.context(), Compose.fill("tomato"), Compose.rectangle(0.75, 0.0, i * timestep * 0.2, 5)),
		(Compose.context(), Compose.fill("gray"), Compose.rectangle(0.75, 0.0, 0.2, 5)))
end

function make_progressbar_2d(s)
	function progressbar_2d(i::Number, timescale::Bool=false, timestep::Number=1, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day")
		if i > 0
			xi = timescale ? i * timestep : i
			if datestart != nothing
				if dateend != nothing
					xi = datestart + ((dateend .- datestart) * (i-1) * timestep)
				else
					xi = datestart + eval(parse(dateincrement))(i-1)
				end
			end
			return Gadfly.plot(s..., Gadfly.layer(xintercept=[xi], Gadfly.Geom.vline(color=["gray"], size=[2Gadfly.pt])))
		else
			return Gadfly.plot(s...)
		end
	end
	return progressbar_2d
end