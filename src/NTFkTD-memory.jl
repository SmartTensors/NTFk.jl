import TensorDecompositions2
import DistributedArrays
import SharedArrays

compose(X::TensorDecompositions2.Tucker{T,N}, modes=collect(1:N)) where {T,N} = TensorDecompositions2.tensorcontractmatrices(TensorDecompositions2.core(X), TensorDecompositions2.factors(X)[modes], modes; transpose=true)

compose(decomp::TensorDecompositions2.CANDECOMP) = TensorDecompositions2.compose(decomp.factors, decomp.lambdas)
@doc """
Composes a full tensor from a decomposition

$(DocumentFunction.documentfunction(compose))
""" compose

@generated function composeshared!(dest::SharedArrays.SharedArray{T,N}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
	quote
		@TensorDecompositions2.nloops $N i dest begin
			elm = zero(T)
			for j in 1:length(lambdas)
				elm += lambdas[j] * (*(@TensorDecompositions2.ntuple($N, k -> factors[k][i_k, j])...))
			end
			@TensorDecompositions2.nref($N, dest, i) = elm
		end
		dest
	end
end

@generated function composeshared(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
	quote
		composeshared!(SharedArrays.SharedArray{T}(@TensorDecompositions2.ntuple $N i -> size(factors[i], 1)), factors, lambdas)
	end
end

composeshared(decomp::TensorDecompositions2.CANDECOMP) = composeshared(decomp.factors, decomp.lambdas)

composeshared!(dest::SharedArrays.SharedArray{T,N}, decomp::TensorDecompositions2.CANDECOMP{T,N}) where {T,N} = composeshared!(dest, decomp.factors, decomp.lambdas)

@generated function composedistributed!(dest::DistributedArrays.DArray{T,N,Array{T,N}}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
	quote
		@TensorDecompositions2.nloops $N i dest begin
			elm = zero(T)
			for j in 1:length(lambdas)
				elm += lambdas[j] * (*(@TensorDecompositions2.ntuple($N, k -> factors[k][i_k, j])...))
			end
			@show elm
			@TensorDecompositions2.nref($N, dest, i) = elm
		end
		dest
	end
end

@generated function composedistributed(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
	quote
		composedistributed!(DistributedArrays.DArray{T,N,Array{T,N}}(undef, @TensorDecompositions2.ntuple $N i -> size(factors[i], 1)), factors, lambdas)
	end
end

composedistributed(decomp::TensorDecompositions2.CANDECOMP) = composedistributed(decomp.factors, decomp.lambdas)

composedistributed!(dest::DistributedArrays.DArray{T,N,Array{T,N}}, decomp::TensorDecompositions2.CANDECOMP{T,N}) where {T,N} = composedistributed!(dest, decomp.factors, decomp.lambdas)