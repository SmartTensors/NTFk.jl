import Gadfly
import Colors
import Compose

function plotmatrix(X::Matrix; minvalue=minimum(X), maxvalue=maximum(X), label="", title="", xlabel="", ylabel="")
	Gadfly.spy(X, Gadfly.Guide.xticks(label=false, ticks=nothing), Gadfly.Guide.yticks(label=false, ticks=nothing), Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Guide.colorkey(label), Gadfly.Scale.ContinuousColorScale(Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red")), minvalue=minvalue, maxvalue=maxvalue), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt))
end

function plottensor(T::TensorDecompositions.Tucker, dim::Integer=1; kw...)
	X = TensorDecompositions.compose(T)
	plottensor(X, dim; kw...)
end

function plottensor(X::Array, dim::Integer=1; minvalue=minimum(X), maxvalue=maximum(X), prefix::String="", movie::Bool=false, title="", hsize=6Compose.inch, vsize=6Compose.inch, moviedir::String=".", quiet::Bool=false, cleanup::Bool=true)
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
		p = plotmatrix(X[nt...], minvalue=minvalue, maxvalue=maxvalue, title=title)
		!quiet && println(framename)
		!quiet && (display(p); println())
		if movie && prefix != ""
			filename = setnewfilename(prefix, i)
			Gadfly.draw(Gadfly.PNG(filename, hsize, vsize), p)
		end
	end
	if movie
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		cleanup && run(`find $moviedir -d 1 -name $prefix-frame*.png -delete`)
	end
end

function plotcmptensor(X1::Array, T2::TensorDecompositions.Tucker, dim::Integer=1; kw...)
	X2 = TensorDecompositions.compose(T2)
	plotcmptensor(X1, X2, dim; kw...)
end

function plotcmptensor(X1::Array, X2::Array, dim::Integer=1; minvalue=minimum([X1 X2]), maxvalue=maximum([X1 X2]), prefix::String="", movie::Bool=false, hsize=12Compose.inch, vsize=6Compose.inch, moviedir::String=".", ltitle::String="True", rtitle::String="Estimated", quiet::Bool=false, cleanup::Bool=true)
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
		p1 = plotmatrix(X1[nt...], minvalue=minvalue, maxvalue=maxvalue, title=ltitle)
		p2 = plotmatrix(X2[nt...], minvalue=minvalue, maxvalue=maxvalue, title=rtitle)
		p = Compose.hstack(p1, p2)
		!quiet && println(framename)
		!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize), p); println())
		if movie && prefix != ""
			filename = setnewfilename(prefix, i)
			Gadfly.draw(Gadfly.PNG(filename, hsize, vsize), p)
		end
	end
	if movie
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		cleanup && run(`find $moviedir -d 1 -name $prefix-frame*.png -delete`)
	end
end

function plotlefttensor(X1::Array, X2::Array, X3::Array, dim::Integer=1; minvalue=minimum([X1 X2 X3]), maxvalue=maximum([X1 X2 X3]), prefix::String="", movie::Bool=false, hsize=24Compose.inch, vsize=6Compose.inch, moviedir::String=".", ltitle::String="True", ctitle::String="Estimated", rtitle::String="Leftover", quiet::Bool=false, cleanup::Bool=true)
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
		p1 = plotmatrix(X1[nt...], minvalue=minvalue, maxvalue=maxvalue, title=ltitle)
		p2 = plotmatrix(X2[nt...], minvalue=minvalue, maxvalue=maxvalue, title=ctitle)
		p3 = plotmatrix(X3[nt...], minvalue=minvalue, maxvalue=maxvalue, title=rtitle)
		p = Compose.hstack(p1, p2, p3)
		!quiet && println(framename)
		!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize), p); println())
		if movie && prefix != ""
			filename = setnewfilename(prefix, i)
			Gadfly.draw(Gadfly.PNG(filename, hsize, vsize), p)
		end
	end
	if movie
		c = `ffmpeg -i $moviedir/$prefix-frame%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		cleanup && run(`find $moviedir -d 1 -name $prefix-frame*.png -delete`)
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
