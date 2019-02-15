import LoadTensorDecompositions
import TensorDecompositions

function loadtucker(f::String, arg...; kw...)
	if isfile(f)
		ans = LoadTensorDecompositions.loadtucker(f, arg...; kw...)
		if ans != nothing
			core, factors, props = ans
			if typeof(core) <: AbstractVector
				n = length(core)
				t = Vector{Any}(undef, 0)
				for i = 1:n
					push!(t, TensorDecompositions.Tucker(factors[i], core[i]))
				end
				return t
			else
				return TensorDecompositions.Tucker(factors, core)
			end
		end
	else
		@warn("File $f does not exist!")
	end
end