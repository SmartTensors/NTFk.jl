import TensorDecompositions

function atensor(X::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP})
	atensor(TensorDecompositions.compose(X))
end
function atensor(X::AbstractArray{T,N}) where {T,N}
	@info("Number of dimensions: $N")
	tsize = size(X)
	mask = Vector{Vector{Bool}}(undef, N)
	for i = 1:N
		@info("D$i ($(tsize[i]))")
		mask[i] = trues(tsize[i])
		for j = 1:tsize[i]
			st = ntuple(k->(k == i ? j : Colon()), N)
			if N == 3
				r = rank(X[st...])
			else
				r = TensorToolbox.mrank(X[st...])
			end
			z = count(X[st...] .> 0)
			println("$j : rank $r non-zeros $z")
			if z == 0
				mask[i][j] = false
			end
			# display(X[st...])
		end
	end
	return mask
end
@doc """
Compute A-tensor rank

$(DocumentFunction.documentfunction(atensor))
""" atensor