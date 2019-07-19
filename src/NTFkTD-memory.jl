import TensorDecompositions
import DistributedArrays
import SharedArrays

compose(X::TensorDecompositions.Tucker{T,N}, modes=collect(1:N)) where {T,N} = TensorDecompositions.tensorcontractmatrices(TensorDecompositions.core(X), TensorDecompositions.factors(X)[modes], modes; transpose=true)

compose(decomp::TensorDecompositions.CANDECOMP) = TensorDecompositions.compose(decomp.factors, decomp.lambdas)
@doc """
Composes a full tensor from a decomposition

$(DocumentFunction.documentfunction(compose))
""" compose

@generated function composeshared!(dest::SharedArrays.SharedArray{T,N}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
	quote
		@TensorDecompositions.nloops $N i dest begin
			elm = zero(T)
			for j in 1:length(lambdas)
				elm += lambdas[j] * (*(@TensorDecompositions.ntuple($N, k -> factors[k][i_k, j])...))
			end
			@TensorDecompositions.nref($N, dest, i) = elm
		end
		dest
	end
end

@generated function composeshared(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
	quote
		composeshared!(SharedArrays.SharedArray{T}(@TensorDecompositions.ntuple $N i -> size(factors[i], 1)), factors, lambdas)
	end
end

composeshared(decomp::TensorDecompositions.CANDECOMP) = composeshared(decomp.factors, decomp.lambdas)

composeshared!(dest::SharedArrays.SharedArray{T,N}, decomp::TensorDecompositions.CANDECOMP{T,N}) where {T,N} = composeshared!(dest, decomp.factors, decomp.lambdas)

@generated function composedistributed!(dest::DistributedArrays.DArray{T,N,Array{T,N}}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
	quote
		@TensorDecompositions.nloops $N i dest begin
			elm = zero(T)
			for j in 1:length(lambdas)
				elm += lambdas[j] * (*(@TensorDecompositions.ntuple($N, k -> factors[k][i_k, j])...))
			end
			@show elm
			@TensorDecompositions.nref($N, dest, i) = elm
		end
		dest
	end
end

@generated function composedistributed(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
	quote
		composedistributed!(DistributedArrays.DArray{T,N,Array{T,N}}(undef, @TensorDecompositions.ntuple $N i -> size(factors[i], 1)), factors, lambdas)
	end
end

composedistributed(decomp::TensorDecompositions.CANDECOMP) = composedistributed(decomp.factors, decomp.lambdas)

composedistributed!(dest::DistributedArrays.DArray{T,N,Array{T,N}}, decomp::TensorDecompositions.CANDECOMP{T,N}) where {T,N} = composedistributed!(dest, decomp.factors, decomp.lambdas)