import Gadfly
import Compose

function plotmatrix(X::Matrix; minvalue=minimum(X), maxvalue=maximum(X), label="")
	Gadfly.spy(X, Gadfly.Guide.xticks(label=false), Gadfly.Guide.yticks(label=false), Gadfly.Guide.xlabel(""), Gadfly.Guide.ylabel(""), Gadfly.Guide.colorkey(label),Gadfly.Scale.ContinuousColorScale(Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red")), minvalue=minvalue, maxvalue=maxvalue))
end

function plottensor(X::Array, dim::Integer=1; minvalue=minimum(X), maxvalue=maximum(X), filename::String="", movie::Bool=false)
	m, n, q = size(X)
	if dim == 1
		for i = 1:m
			label = "Row $i"
			p = plotmatrix(X[i,:,:], minvalue=minvalue, maxvalue=maxvalue, label=label)
			println(label)
			display(p); println()
		end
	elseif dim == 2
		for i = 1:n
			label = "Column $i"
			p = plotmatrix(X[:,i,:], minvalue=minvalue, maxvalue=maxvalue, label=label)
			println(label)
			display(p); println()
		end
	elseif dim == 3
		for i = 1:q
			label = "Layer $i"
			p = plotmatrix(X[:,:,i], minvalue=minvalue, maxvalue=maxvalue, label=label)
			println(label)
			display(p); println()
		end
	end

	if movie
		filename = setnewfilename(filename, frame)
	end
	if filename != ""
		Base.display(Images.load(filename))
	end
end

function plotcmptensor(X1::Array, X2::Array, dim::Integer=1; minvalue=minimum([X1 X2]), maxvalue=maximum([X1 X2]), filename::String="", movie::Bool=false)
	m, n, q = size(X1)
	if dim == 1
		for i = 1:m
			label = "Row $i"
			p1 = plotmatrix(X1[i,:,:], minvalue=minvalue, maxvalue=maxvalue, label=label)
			p2 = plotmatrix(X2[i,:,:], minvalue=minvalue, maxvalue=maxvalue, label=label)
			p = Gadfly.hstack(p1, p2)
			println(label)
			Compose.draw(Compose.PNG(24Compose.cm, 6Compose.cm),p); println()
		end
	elseif dim == 2
		for i = 1:n
			label = "Column $i"
			p1 = plotmatrix(X1[:,i,:], minvalue=minvalue, maxvalue=maxvalue, label=label)
			p2 = plotmatrix(X2[:,i,:], minvalue=minvalue, maxvalue=maxvalue, label=label)
			p = Gadfly.hstack(p1, p2)
			println(label)
			Compose.draw(Compose.PNG(24Compose.cm, 6Compose.cm),p); println()
		end
	elseif dim == 3
		for i = 1:q
			label = "Layer $i"
			p1 = plotmatrix(X1[:,:,i], minvalue=minvalue, maxvalue=maxvalue, label=label)
			p2 = plotmatrix(X2[:,:,i], minvalue=minvalue, maxvalue=maxvalue, label=label)
			p = Gadfly.hstack(p1, p2)
			println(label)
			Compose.draw(Compose.PNG(24Compose.cm, 6Compose.cm),p); println()
		end
	end

	if movie
		filename = setnewfilename(filename, frame)
	end
	if filename != ""
		Base.display(Images.load(filename))
	end
end
