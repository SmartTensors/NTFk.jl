import TensorDecompositions
import DistributedArrays

@generated function composeshared!(dest::SharedArray{T,N}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T}) where {T,N}
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
		composeshared!(SharedArray{T}(@TensorDecompositions.ntuple $N i -> size(factors[i], 1)), factors, lambdas)
	end
end

composeshared(decomp::TensorDecompositions.CANDECOMP) = composedistributed(decomp.factors, decomp.lambdas)

composeshared!(dest::SharedArray{T,N}, decomp::TensorDecompositions.CANDECOMP{T,N}) where {T,N} = composedistributed!(dest, decomp.factors, decomp.lambdas)

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
		composedistributed!(DistributedArrays.DArray{T,N,Array{T,N}}(@TensorDecompositions.ntuple $N i -> size(factors[i], 1)), factors, lambdas)
	end
end

composedistributed(decomp::TensorDecompositions.CANDECOMP) = composedistributed(decomp.factors, decomp.lambdas)

composedistributed!(dest::DistributedArrays.DArray{T,N,Array{T,N}}, decomp::TensorDecompositions.CANDECOMP{T,N}) where {T,N} = composedistributed!(dest, decomp.factors, decomp.lambdas)

