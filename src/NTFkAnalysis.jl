function analysis(X::AbstractArray{T,N}, dsizes::Vector{Int64}, dim, nTF; kw...) where {T,N}
	csize = collect(size(X))
	ndimensons = length(csize)
	sizes = Vector{Tuple}(undef, 0)
	for i in dsizes
		nt = ntuple(k->(k == dim ? i : csize[k]), ndimensons)
		push!(sizes, nt)
	end
	@info("Sizes: $(sizes)")
	analysis(X, sizes, nTF; kw...)
end