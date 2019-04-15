import LoadTensorDecompositions
import TensorDecompositions
import FileIO
import JLD
import JLD2

function loadtucker(f::String, arg...; kw...)
	if !isfile(f)
		@warn("File $f does not exist!")
	end
	try
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
		else
			@warn("There is a problem reading file $f!")
			return nothing
		end
	catch
		return localloadtucker(f::String, arg...; kw...)
	end
end

function localloadtucker(f::String, arg...; kw...)
	d = FileIO.load(f)
	if [keys(d)...][1] == "tucker_vector"
		di = d["tucker_vector"]
		n = length(di)
		t = Vector{Any}(undef, 0)
		for i = 1:n
			push!(t, TensorDecompositions.Tucker(di[i].factors, di[i].core))
		end
		return t
	else
		di = d["tucker"]
		return TensorDecompositions.Tucker(di.factors, di.core)
	end
end

function savetucker(tucker_spnn, f::String)
	if typeof(tucker_spnn) <: TensorDecompositions.Tucker
		FileIO.save(f, "tucker", tucker_spnn)
	else
		FileIO.save(f, "tucker_vector", tucker_spnn)
	end
end