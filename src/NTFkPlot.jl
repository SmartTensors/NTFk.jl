import Gadfly
import Measures
import Colors
import Compose
import TensorToolbox
import TensorDecompositions

function plotfactor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1, cutoff::Number=0; kw...)
	NMFk.plotmatrix(getfactor(t, dim, cutoff); kw...)
end

function plotfactors(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, cutoff::Number=0; prefix="", kw...)
	recursivemkdir(prefix; filename=true)
	for i = eachindex(t.factors)
		display(plotfactor(t, i, cutoff; filename="$(prefix)_factor$(i).png", kw...))
		println()
	end
end

function plotcore(t::TensorDecompositions.Tucker, dim::Integer=1, cutoff::Number=0; kw...)
	plottensor(t.core, dim; progressbar=nothing, cutoff=true, cutvalue=cutoff, kw...)
end

function plottensor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; mask=nothing, transform=nothing, kw...)
	X = TensorDecompositions.compose(t)
	if !isnothing(transform)
		X = transform.(X)
	end
	if typeof(mask) <: Number
		nanmask!(X, mask)
	else
		nanmask!(X, mask, dim)
	end
	plottensor(X, dim; kw...)
end

function plottensor(X::AbstractArray{T,N}, dim::Integer=1; permute::Bool=false, mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=NMFk.minimumnan(X), maxvalue=NMFk.maximumnan(X), prefix::AbstractString="", keyword="frame", movie::Bool=false, title="", hsize::Measures.AbsoluteLength=6Compose.inch, vsize::Measures.AbsoluteLength=6Compose.inch, dpi::Integer=imagedpi, moviedir::AbstractString=".", quiet::Bool=false, cleanup::Bool=true, sizes=size(X), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=(!isnothing(datestart)) ? datestart + Core.eval(Main, Meta.parse(dateincrement))(sizes[dim]) : nothing, progressbar=progressbar_regular, colormap=colormap_gyr, cutoff::Bool=false, cutvalue::Number=0, vspeed=1.0, movieformat="mp4", movieopacity::Bool=false, kw...) where {T <: Number, N}
	if !checkdimension(dim, N)
		return
	end
	recursivemkdir(moviedir; filename=false)
	recursivemkdir(prefix; filename=true)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		M = permute ? permutedims(X[nt...]) : X[nt...]
		if cutoff && NMFk.maximumnan(M) .<= cutvalue
			continue
		end
		g = NMFk.plotmatrix(M; minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap, plot=true, quiet=true, kw...)
		if !isnothing(progressbar)
			f = progressbar(i, timescale, timestep, datestart, dateend, dateincrement)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		!quiet && ((sizes[dim] > 1) && println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(g, f)); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Compose.vstack(g, f))
		end
	end
	if movie && prefix != ""
		makemovie(movieformat=movieformat, movieopacity=movieopacity, moviedir=moviedir, prefix=prefix, keyword=keyword, cleanup=cleanup, quiet=quiet, vspeed=vspeed)
	end
end

function plot2matrices(X1::AbstractMatrix, X2::AbstractMatrix; kw...)
	plot2tensors([X1], [X2], 1; minvalue=NMFk.minimumnan([X1 X2]), maxvalue=NMFk.maximumnan([X1 X2]), kw...)
end

function plot2tensors(X1::Array{T,N}, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...) where {T <: Number, N}
	X2 = convert(Array{T,N}, TensorDecompositions.compose(T2))
	plot2tensors(X1, X2, dim; kw...)
end

function plot2tensors(X1::AbstractArray{T,N}, X2::AbstractArray{T,N}, dim::Integer=1; mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=NMFk.minimumnan([X1 X2]), maxvalue=NMFk.maximumnan([X1 X2]), minvalue2=minvalue, maxvalue2=maxvalue, movie::Bool=false, hsize::Measures.AbsoluteLength=12Compose.inch, vsize::Measures.AbsoluteLength=6Compose.inch, dpi::Integer=imagedpi, title::AbstractString="", moviedir::AbstractString=".", prefix::AbstractString = "", keyword="frame", ltitle::AbstractString="", rtitle::AbstractString="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=(!isnothing(datestart)) ? datestart + Core.eval(Main, Meta.parse(dateincrement))(sizes[dim]) : nothing, dateincrement::AbstractString="Dates.Day", progressbar=progressbar_regular, uniformscaling::Bool=true, colormap=colormap_gyr, vspeed=1.0, movieformat="mp4", movieopacity::Bool=false, kw...) where {T <: Number, N}
	if !checkdimension(dim, N)
		return
	end
	recursivemkdir(prefix; filename=true)
	if !uniformscaling
		minvalue = NMFk.minimumnan(X1)
		maxvalue = NMFk.maximumnan(X1)
		minvalue2 = NMFk.minimumnan(X2)
		maxvalue2 = NMFk.maximumnan(X2)
	end
	recursivemkdir(moviedir; filename=false)
	@assert sizes == size(X2)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		g1 = NMFk.plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=colormap, quiet=true, plot=true, kw...)
		g2 = NMFk.plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, title=rtitle, colormap=colormap, quiet=true, plot=true, kw...)
		if title != ""
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0.0001Compose.h),
				(Compose.context(), Compose.fill("gray"), Compose.fontsize(20Compose.pt), Compose.text(0.5Compose.w, 0, title * " : " * sprintf("%06d", i), Compose.hcenter, Compose.vtop)))
		else
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		if !isnothing(progressbar)
			f = progressbar(i, timescale, timestep, datestart, dateend, dateincrement)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		!quiet && (sizes[dim] > 1) && println(framename)
		!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(t, Compose.hstack(g1, g2), f)); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Compose.vstack(t, Compose.hstack(g1, g2), f))
		end
	end
	if movie && prefix != ""
		makemovie(movieformat=movieformat, movieopacity=movieopacity, moviedir=moviedir, prefix=prefix, keyword=keyword, cleanup=cleanup, quiet=quiet, vspeed=vspeed)
	end
end

function plot3matrices(X1::AbstractMatrix, X2::AbstractMatrix, X3::AbstractMatrix; kw...)
	plot3tensors([X1], [X2], [X3], 1; minvalue=NMFk.minimumnan([X1 X2 X3]), maxvalue=NMFk.maximumnan([X1 X2 X3]), kw...)
end

function plotcmptensors(X1::Array{T,N}, X2::Array{T,N}, dim::Integer=1; center=true, transform=nothing, mask=nothing, kw...) where {T <: Number, N}
	plot2tensors(X1, X2, dim; minvalue=NMFk.minimumnan([X1 X2]), maxvalue=NMFk.maximumnan([X1 X2]), kw...)
end

function plotcmptensors(X1::Array{T,N}, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; center=true, transform=nothing, mask=nothing, kw...) where {T <: Number, N}
	X2 = TensorDecompositions.compose(T2)
	if !isnothing(transform)
		X2 = transform.(X2)
	end
	nanmask!(X2, mask)
	plot2tensors(X1, convert(Array{T,N}, X2), dim; minvalue=NMFk.minimumnan([X1 X2]), maxvalue=NMFk.maximumnan([X1 X2]), kw...)
end

function plot3tensors(X1::AbstractArray{T,N}, X2::AbstractArray{T,N}, X3::AbstractArray{T,N}, dim::Integer=1; mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=NMFk.minimumnan([X1 X2 X3]), maxvalue=NMFk.maximumnan([X1 X2 X3]), minvalue2=minvalue, maxvalue2=maxvalue, minvalue3=minvalue, maxvalue3=maxvalue, prefix::AbstractString="", keyword="frame", movie::Bool=false, hsize::Measures.AbsoluteLength=24Compose.inch, vsize::Measures.AbsoluteLength=6Compose.inch, dpi::Integer=imagedpi,moviedir::AbstractString=".", ltitle::AbstractString="", ctitle::AbstractString="", rtitle::AbstractString="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=nothing, progressbar=progressbar_regular, barratio::Number=1/2, uniformscaling::Bool=true, vspeed=1.0, movieformat="mp4", movieopacity::Bool=false, colormap=colormap_gyr, overlap::Bool=false, key_label_font_size=12Gadfly.pt, opacity=0.8, gla=[], signalnames=["T1", "T2", "T3"], kw...) where {T <: Number, N}
	recursivemkdir(prefix; filename=true)
	if !checkdimension(dim, N)
		return
	end
	if !uniformscaling
		minvalue = NMFk.minimumnan(X1)
		maxvalue = NMFk.maximumnan(X1)
		minvalue2 = NMFk.minimumnan(X2)
		maxvalue2 = NMFk.maximumnan(X2)
		minvalue3 = NMFk.minimumnan(X3)
		maxvalue3 = NMFk.maximumnan(X3)
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
	if length(gla) > 1
		@assert length(gla) == 3
	else
		gla = Vector{Any}(undef, 3)
		for i = 1:3
			gla[i] = []
		end
	end
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i / $(sizes[dim])"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		if overlap
			g1 = NMFk.plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, key_label_font_size=key_label_font_size, title=ltitle, colormap=nothing, defaultcolor=Colors.RGBA(1.0,0.0,0.0,opacity), code=true, gl=gla[1], kw...)
			g2 = NMFk.plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, key_label_font_size=key_label_font_size, title=ctitle, colormap=nothing, defaultcolor=Colors.RGBA(0.0,0.0,1.0,opacity), code=true, gl=gla[2], kw...)
			g3 = NMFk.plotmatrix(X3[nt...]; minvalue=minvalue3, maxvalue=maxvalue3, key_label_font_size=key_label_font_size, title=rtitle, colormap=nothing, defaultcolor=Colors.RGBA(0.0,0.502,0.0,opacity), code=true, gl=gla[3], kw...)
			g = Compose.hstack(Gadfly.plot(g1..., g2..., g3..., Gadfly.Guide.manual_color_key("", signalnames, ["red", "blue", "green"])))
		else
			g1 = NMFk.plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=colormap1, key_label_font_size=key_label_font_size, gl=gla[1], plot=true, quiet=true, kw...)
			g2 = NMFk.plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, title=ctitle, colormap=colormap2, key_label_font_size=key_label_font_size, gl=gla[2], plot=true, quiet=true, kw...)
			g3 = NMFk.plotmatrix(X3[nt...]; minvalue=minvalue3, maxvalue=maxvalue3, title=rtitle, colormap=colormap3, key_label_font_size=key_label_font_size, gl=gla[3], plot=true, quiet=true, kw...)
			g = Compose.hstack(g1, g2, g3)
		end
		if !isnothing(progressbar)
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
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), Compose.hstack(g1, g2, g3)), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f)))); println()
				end
			else
				if overlap
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(g, f)); println()
				else
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(Compose.hstack(g1, g2, g3), f)); println()
				end
			end
		end
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			if typeof(f) != Compose.Context
				if overlap
					Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), g), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f))))
				else
					Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), Compose.hstack(g1, g2, g3)), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f))))
				end
			else
				if overlap
					Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Compose.vstack(g, f))
				else
					Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Compose.vstack(Compose.hstack(g1, g2, g3), f))
				end
			end
		end
	end
	if movie && prefix != ""
		return makemovie(movieformat=movieformat, movieopacity=movieopacity, moviedir=moviedir, prefix=prefix, keyword=keyword, cleanup=cleanup, quiet=quiet, vspeed=vspeed)
	end
end

function plotMtensors(X::AbstractVector{Array{T,N}}, dim::Integer=1; sizes=size(X[1]), M=length(X), mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=NMFk.minimumnan(map(i->NMFk.minimumnan(X[i]), 1:M)), maxvalue=NMFk.maximumnan(map(i->NMFk.maximumnan(X[i]), 1:M)), prefix::AbstractString="", keyword="frame", movie::Bool=false, hsize::Measures.AbsoluteLength=24Compose.inch, vsize::Measures.AbsoluteLength=6Compose.inch, dpi::Integer=imagedpi,moviedir::AbstractString=".", ltitle::AbstractString="", ctitle::AbstractString="", rtitle::AbstractString="", quiet::Bool=false, cleanup::Bool=true, timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateincrement::AbstractString="Dates.Day", dateend=nothing, progressbar=progressbar_regular, barratio::Number=1/2, uniformscaling::Bool=true, vspeed=1.0, movieformat="mp4", movieopacity::Bool=false, colormap=colormap_gyr, overlap::Bool=false, key_label_font_size=16Gadfly.pt, opacity=0.8, gla=[], signalnames=["T$i" for i=1:M], signalcolors=colors[1:M], kw...) where {T <: Number, N}
	recursivemkdir(prefix; filename=true)
	if !checkdimension(dim, N)
		return
	end
	recursivemkdir(moviedir; filename=false)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	if length(gla) > 1
		@assert length(gla) == M
	else
		gla = Vector{Any}(undef, M)
		for i = 1:M
			gla[i] = []
		end
	end
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i / $(sizes[dim])"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		gv = Vector{Any}(undef, M)
		if overlap
			for m = 1:M
				gv[m] = NMFk.plotmatrix(X[m][nt...]; minvalue=minvalue, maxvalue=maxvalue, key_label_font_size=key_label_font_size, title=ltitle, colormap=nothing, defaultcolor=Colors.RGBA(parse.(Colors.Colorant, colors[m]), opacity), code=true, kw...)
			end
			g = Compose.hstack(Gadfly.plot(vcat(map(x->[x...], gv)...)..., Gadfly.Guide.manual_color_key("", signalnames, signalcolors)))
		else
			for m = 1:M
				if length(colormap) > 1
					gv[m] = NMFk.plotmatrix(X[m][nt...]; minvalue=minvalue, maxvalue=maxvalue, key_label_font_size=key_label_font_size, title=ltitle, colormap=colormap[m], plot=true, quiet=true, kw...)
				else
					gv[m] = NMFk.plotmatrix(X[m][nt...]; minvalue=minvalue, maxvalue=maxvalue, key_label_font_size=key_label_font_size, title=ltitle, colormap=colormap, plot=true, quiet=true, kw...)
				end
			end
			g = Compose.hstack(gv...)
		end
		if !isnothing(progressbar)
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
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), Compose.hstack(gv...)), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f)))); println()
				end
			else
				if overlap
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(g, f)); println()
				else
					Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(Compose.hstack(gv...), f)); println()
				end
			end
		end
		if prefix != ""
			if sizes[dim] > 1
				filename = setnewfilename(prefix, i; keyword=keyword)
			else
				filename = prefix * ".png"
			end
			if typeof(f) != Compose.Context
				if overlap
					Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), g), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f))))
				else
					Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), Compose.hstack(gv...)), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f))))
				end
			else
				if overlap
					Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Compose.vstack(g, f))
				else
					Gadfly.draw(Gadfly.PNG(NMFk.joinpathcheck(moviedir, filename), hsize, vsize, dpi=dpi), Compose.vstack(Compose.hstack(gv...), f))
				end
			end
		end
	end
	if movie && prefix != ""
		makemovie(movieformat=movieformat, movieopacity=movieopacity, moviedir=moviedir, prefix=prefix, keyword=keyword, cleanup=cleanup, quiet=quiet, vspeed=vspeed)
	end
end

function plotleftmatrix(X1::AbstractMatrix, X2::AbstractMatrix; minvalue=NMFk.minimumnan([X1 X2]), maxvalue=NMFk.maximumnan([X1 X2]), minvalue3=nothing, maxvalue3=nothing, center=true, kw...)
	D = X2 .- X1
	minvalue3 = isnothing(minvalue3) ? NMFk.minimumnan(D) : minvalue3
	maxvalue3 = isnothing(maxvalue3) ? NMFk.maximumnan(D) : maxvalue3
	if center
		minvalue3, maxvalue3 = min(minvalue3, -maxvalue3), max(maxvalue3, -minvalue3)
	end
	plot3tensors([X1], [X2], [D], 1; minvalue=NMFk.minimumnan([X1 X2]), maxvalue=NMFk.maximumnan([X1 X2]), minvalue3=minvalue3, maxvalue3=maxvalue3, kw...)
end

function plotlefttensor(X1::Array, X2::Array, dim::Integer=1; minvalue=NMFk.minimumnan([X1 X2]), maxvalue=NMFk.maximumnan([X1 X2]), minvalue3=nothing, maxvalue3=nothing, center=true, kw...)
	D = X2 .- X1
	minvalue3 = isnothing(minvalue3) ? NMFk.minimumnan(D) : minvalue3
	maxvalue3 = isnothing(maxvalue3) ? NMFk.maximumnan(D) : maxvalue3
	if center
		minvalue3, maxvalue3 = min(minvalue3, -maxvalue3), max(maxvalue3, -minvalue3)
	end
	plot3tensors(X1, X2, D, dim; minvalue=minvalue, maxvalue=maxvalue, minvalue3=minvalue3, maxvalue3=maxvalue3, kw...)
end

function plotlefttensor(X1::Array{T,N}, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; minvalue=nothing, maxvalue=nothing, minvalue3=nothing, maxvalue3=nothing, center=true, transform=nothing, mask=nothing, kw...) where {T <: Number, N}
	X2 = TensorDecompositions.compose(T2)
	if !isnothing(transform)
		X2 = transform.(X2)
	end
	D = X2 - X1
	nanmask!(X2, mask)
	nanmask!(D, mask)
	minvalue = isnothing(minvalue) ? NMFk.minimumnan([X1 X2]) : minvalue
	maxvalue = isnothing(maxvalue) ? NMFk.maximumnan([X1 X2]) : maxvalue
	minvalue3 = isnothing(minvalue3) ? NMFk.minimumnan(D) : minvalue3
	maxvalue3 = isnothing(maxvalue3) ? NMFk.maximumnan(D) : maxvalue3
	if center
		minvalue3, maxvalue3 = min(minvalue3, -maxvalue3), max(maxvalue3, -minvalue3)
	end
	plot3tensors(X1, convert(Array{T,N}, X2), convert(Array{T,N}, D), dim; minvalue=minvalue, maxvalue=maxvalue, minvalue3=minvalue3, maxvalue3=maxvalue3, kw...)
end