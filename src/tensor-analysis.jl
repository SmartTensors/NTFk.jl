import Gadfly
import Colors
import Compose

function atensor(X::Array)
	nd = ndims(X)
	info("Number of dimensions: $nd")
	tsize = size(X)
	for i = 1:nd
		info("D$i ($(tsize[i]))")
		for j = 1:tsize[i]
			st = ntuple(k->(k == i ? j : :), 3)
			r = rank(X[st...])
			z = count(X[st...] .> 0)
			println("$j : rank $r non-zeros $z")
			# display(X[st...])
		end
	end
end