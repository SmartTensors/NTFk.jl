import LoadTensorDecompositions
import TensorDecompositions

function loadtucker(f::String, arg...; kw...)
	core, factors, props = LoadTensorDecompositions.loadtucker(f, arg...; kw...)
	return TensorDecompositions.Tucker(factors, core)
end