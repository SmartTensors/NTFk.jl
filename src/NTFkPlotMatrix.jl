import Gadfly
import Measures
import Colors
import Compose

function plotmatrix(X::AbstractVector; kw...)
	plotmatrix(convert(Array{Float64,2}, permutedims(X)); kw...)
end

function plotmatrix(X::AbstractMatrix; minvalue=minimumnan(X), maxvalue=maximumnan(X), label="", title="", xlabel="", ylabel="", xticks=nothing, yticks=nothing, xgrid=nothing, ygrid=nothing, gm=[Gadfly.Guide.xticks(label=false, ticks=nothing), Gadfly.Guide.yticks(label=false, ticks=nothing)], masize::Int64=0, colormap=colormap_gyr, filename::String="", hsize=6Compose.inch, vsize=6Compose.inch, figuredir::String=".", colorkey::Bool=true, mask=nothing, polygon=nothing, contour=nothing, linewidth::Measures.Length{:mm,Float64}=1Gadfly.pt, linecolor="gray", defaultcolor=nothing, pointsize=1Gadfly.pt, transform=nothing, code=false, nbins::Integer=0)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	Xp = deepcopy(min.(max.(movingaverage(X, masize), minvalue), maxvalue))
	if transform != nothing
		Xp = transform.(Xp)
	end
	nanmask!(Xp, mask)
	is, js, xs = Gadfly._findnz(x->!isnan(x), Xp)
	n, m = size(Xp)
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
	gt = [Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt, bar_spacing=0Gadfly.mm), Gadfly.Scale.x_continuous, Gadfly.Scale.y_continuous, Gadfly.Coord.cartesian(yflip=true, fixed=true, xmin=0.5, xmax=m+.5, ymin=0.5, ymax=n+.5)]
	if defaultcolor == nothing
		l = [Gadfly.layer(x=js, y=is, color=xs, Gadfly.Geom.rectbin())]
	else
		if nbins == 0
			l = [Gadfly.layer(x=js, y=is, Gadfly.Theme(default_color=defaultcolor, point_size=pointsize, highlight_width=0Gadfly.pt))]
		else
			l = []
			s = (maxvalue - minvalue) / nbins
			s1 = minvalue
			s2 = minvalue + s
			for i = 1:nbins
				id = find(i->(i > s1 && i <= s2), xs)
				c = Colors.RGBA(defaultcolor.r, defaultcolor.g, defaultcolor.b, defaultcolor.alpha/i)
				sum(id) > 0 && (l = [l..., Gadfly.layer(x=js[id], y=is[id], Gadfly.Theme(default_color=c, point_size=pointsize, highlight_width=0Gadfly.pt))])
				s1 += s
				s2 += s
			end
		end
	end
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