import TensorDecompositions
import DistributedArrays

@generated function composeshared!{T,N}(dest::SharedArray{T,N}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T})
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

@generated function composeshared{T,N}(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T})
	quote
		composeshared!(SharedArray{T}(@TensorDecompositions.ntuple $N i -> size(factors[i], 1)), factors, lambdas)
	end
end

composeshared(decomp::TensorDecompositions.CANDECOMP) = composedistributed(decomp.factors, decomp.lambdas)

composeshared!{T,N}(dest::SharedArray{T,N}, decomp::TensorDecompositions.CANDECOMP{T,N}) = composedistributed!(dest, decomp.factors, decomp.lambdas)

@generated function composedistributed!{T,N}(dest::DistributedArrays.DArray{T,N,Array{T,N}}, factors::NTuple{N, Matrix{T}}, lambdas::Vector{T})
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

@generated function composedistributed{T,N}(factors::NTuple{N, Matrix{T}}, lambdas::Vector{T})
	quote
		composedistributed!(DistributedArrays.DArray{T,N,Array{T,N}}(@TensorDecompositions.ntuple $N i -> size(factors[i], 1)), factors, lambdas)
	end
end

composedistributed(decomp::TensorDecompositions.CANDECOMP) = composedistributed(decomp.factors, decomp.lambdas)

composedistributed!{T,N}(dest::DistributedArrays.DArray{T,N,Array{T,N}}, decomp::TensorDecompositions.CANDECOMP{T,N}) = composedistributed!(dest, decomp.factors, decomp.lambdas)

