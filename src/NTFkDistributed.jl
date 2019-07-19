import DistributedArrays

"""
Distribute

$(DocumentFunction.documentfunction(distribute))
"""
function distribute(X::AbstractArray{T,N}) where {T,N}
	dX = DistributedArrays.distribute(X)
	sz = size(X)
	@info "Dimension 1: Distributing array with size $(sz)"
	for i = 2:N
		a = [i for i = 1:N]
		k = a .!= i
		v = [i, a[k]...]
		vb = "dX$i"
		@info "Dimension $i: Distributing array with size $(sz[v])"
		@eval(global $(Symbol(vb)) = DistributedArrays.distribute(permutedims($X, $v)))
	end
	return dX
end