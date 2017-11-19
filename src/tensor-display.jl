import Gadfly
import Colors
import Compose
import TensorToolbox
import TensorDecompositions

function plotmatrix(X::Matrix; minvalue=minimum(X), maxvalue=maximum(X), label="", title="", xlabel="", ylabel="")
	Gadfly.spy(X, Gadfly.Guide.xticks(label=false, ticks=nothing), Gadfly.Guide.yticks(label=false, ticks=nothing), Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Guide.colorkey(label), Gadfly.Scale.ContinuousColorScale(Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red")), minvalue=minvalue, maxvalue=maxvalue), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt))
end

function plottensor(T::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...)
	X = TensorDecompositions.compose(T)
	plottensor(X, dim; kw...)
end

function plottensor(X::Array, dim::Integer=1; minvalue=minimum(X), maxvalue=maximum(X), prefix::String="", movie::Bool=false, title="", hsize=6Compose.inch, vsize=6Compose.inch, moviedir::String=".", quiet::Bool=false, cleanup::Bool=true, timestep::Number=0.001, progressbar::Bool=true)
	sizes = size(X)
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	ndimensons = length(sizes)
	if dim > ndimensons || dim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	if ndimensons <= 3
		dimname = ("Row", "Column", "Layer")
	else
		dimname = ntuple(i->("D$i"), ndimensons)
	end
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : :), ndimensons)
		g = plotmatrix(X[nt...], minvalue=minvalue, maxvalue=maxvalue, title=title)
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
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize), p)
		end
	end
	if movie
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
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

function namedimension(ndimensons::Int)
	if ndimensons <= 3
		dimname = ("T", "X", "Y")
	else
		dimname = ntuple(i->("C$i"), ndimensons)
	end
end

function plottensorcomponents(X1::Array, t2::TensorDecompositions.CANDECOMP; prefix::String="", filter=(), kw...)
	ndimensons = length(size(X1))
	crank = length(t2.lambdas)
	for i = 1:crank
		info("Making component $i movie ...")
		ntt = deepcopy(t2)
		ntt.lambdas[1:end .!== i] = 0
		if length(filter) == 0
			X2 = TensorDecompositions.compose(ntt)
		else
			X2 = TensorDecompositions.compose(ntt)[filter...]
		end
		dNTF.plotcmptensor(X1, X2; progressbar=false, prefix=prefix * string(i), kw...)
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
	for i = 1:crank
		info("Making component $(dimname[dim])-$i movie ...")
		ntt = deepcopy(t2)
		for j = 1:crank
			if i !== j
				nt = ntuple(k->(k == dim ? j : :), ndimensons)
				ntt.core[nt...] .= 0
			end
		end
		if length(filter) == 0
			X2 = TensorDecompositions.compose(ntt)
		else
			X2 = TensorDecompositions.compose(ntt)[filter...]
		end
		title = pdim > 1 ? "$(dimname[dim])-$i" : ""
		dNTF.plotcmptensor(permutedims(X1, pt), permutedims(X2, pt); progressbar=false, title=title, prefix=prefix * string(i),  kw...)
	end
end

function plot2tensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t2.core), prefix::String="", filter=(), kw...)
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
	X2= Vector{Any}(crank)
	for i = 1:crank
		info("Making component $(dimname[dim])-$i movie ...")
		ntt = deepcopy(t2)
		for j = 1:crank
			if i !== j
				nt = ntuple(k->(k == dim ? j : :), ndimensons)
				ntt.core[nt...] .= 0
			end
		end
		if length(filter) == 0
			X2[i] = TensorDecompositions.compose(ntt)
		else
			X2[i] = TensorDecompositions.compose(ntt)[filter...]
		end
	end
	dNTF.plotlefttensor(permutedims(X1, pt), permutedims(X2[1], pt), permutedims(X2[2], pt); progressbar=false, prefix=prefix, kw...)
end

function plotcmptensor(X1::Array, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...)
	X2 = TensorDecompositions.compose(T2)
	plotcmptensor(X1, X2, dim; kw...)
end

function plotcmptensor(X1::Array, X2::Array, dim::Integer=1; minvalue=minimum([X1 X2]), maxvalue=maximum([X1 X2]), prefix::String="", movie::Bool=false, hsize=12Compose.inch, vsize=6Compose.inch, title::String="", moviedir::String=".", ltitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, timestep::Number=0.001, progressbar::Bool=true)
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	sizes = size(X1)
	@assert sizes == size(X2)
	ndimensons = length(sizes)
	if dim > ndimensons || dim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	if ndimensons <= 3
		dimname = ("Row", "Column", "Layer")
	else
		dimname = ntuple(i->("D$i"), ndimensons)
	end
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : :), ndimensons)
		g1 = plotmatrix(X1[nt...], minvalue=minvalue, maxvalue=maxvalue, title=ltitle)
		g2 = plotmatrix(X2[nt...], minvalue=minvalue, maxvalue=maxvalue, title=rtitle)
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
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize), p)
		end
	end
	if movie
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		cleanup && run(`find $moviedir -name $prefix-"frame*.png" -delete`)
	end
end

function plotlefttensor(X1::Array, X2::Array, X3::Array, dim::Integer=1; minvalue=minimum([X1 X2 X3]), maxvalue=maximum([X1 X2 X3]), prefix::String="", movie::Bool=false, hsize=24Compose.inch, vsize=6Compose.inch, moviedir::String=".", ltitle::String="", ctitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, timestep::Number=0.001, progressbar::Bool=true)
	if !isdir(moviedir)
		mkdir(moviedir)
	end
	sizes = size(X1)
	@assert sizes == size(X2)
	ndimensons = length(sizes)
	if dim > ndimensons || dim < 1
		warn("Dimension should be >=1 or <=$(length(sizes))")
		return
	end
	if ndimensons <= 3
		dimname = ("Row", "Column", "Layer")
	else
		dimname = ntuple(i->("D$i"), ndimensons)
	end
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : :), ndimensons)
		g1 = plotmatrix(X1[nt...], minvalue=minvalue, maxvalue=maxvalue, title=ltitle)
		g2 = plotmatrix(X2[nt...], minvalue=minvalue, maxvalue=maxvalue, title=ctitle)
		g3 = plotmatrix(X3[nt...], minvalue=minvalue, maxvalue=maxvalue, title=rtitle)
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
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize), p)
		end
	end
	if movie
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		cleanup && run(`find $moviedir -name $prefix-"frame*.png" -delete`)
	end
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

"Convert `@sprintf` macro into `sprintf` function"
sprintf(args...) = eval(:@sprintf($(args...)))
