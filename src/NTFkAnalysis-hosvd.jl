import TensorDecompositions
import TensorToolbox

"""
High-order singular value decomposition (HO-SVD).
"""
function hosvd(tensor::StridedArray{T,N}, core_dims::NTuple{N, Int}, eigmethod=trues(N), eigreduce=eigmethod; order=1:N, pad_zeros::Bool=false, compute_error::Bool=true, compute_rank::Bool=true, which=:LM, tol=0.0, maxiter=300, rtol=0.) where {T,N}
	pad_zeros || _check_tensor(tensor, core_dims)

	csize = size(tensor)

	factors = map(order) do i
		X = _col_unfold(tensor, i)
		if eigmethod[i] || (core_dims[i] < csize[i])
			nev = core_dims[i]
			e = nothing
			while true
				e = eigs(X'X, nev=nev; which=which, tol=tol, maxiter=maxiter)
				# f = eigen(Symmetric(X'X), max(1, size(X,2)-core_dims[i]+1):size(X,2)).vectors
				r = maximum(abs.(e[6]))
				if nev == 1 || r > rtol || !eigreduce[i]
					break
				end
				nev -= 1
			end
			if compute_error
				info("D$i components $(core_dims[i])->$(e[3]) errors: max $(maximum(e[6])) min $(minimum(e[6])) iterations $(e[4])")
			end
			f = e[2]
		else
			f = eig(X'X)[2]
			if compute_error
				info("D$i components $(core_dims[i])")
			end
		end
		if pad_zeros && size(f, 2) < core_dims[i] # fill missing factors with zeros
			warn("Zero slices ($(core_dims[i]-size(f, 2))) added in dimension $i ")
			f = hcat(f, zeros(T, size(tensor, i), core_dims[i]-size(f, 2)))
		end
		mapslices(_check_sign, f, 1)
	end

	res = TensorDecompositions.Tucker((factors[order]...,), tensorcontractmatrices(tensor, factors[order]))
	if compute_error
		TensorDecompositions._set_rel_residue(res, tensor)
		info("Error: $(res.props[:rel_residue])")
		info("Vector Norm: $(vecnorm(tensor .- compose(res)))")
	end
	if compute_rank
		info("HOSVD core rank: $(TensorToolbox.mrank(res.core))")
	end
	return res
end

hosvd(tensor::StridedArray{T,N}, r::Int; compute_error::Bool=false, pad_zeros::Bool=false) where {T,N} =
	NTFk.hosvd(tensor, (fill(r, N)...,); compute_error=compute_error, pad_zeros=pad_zeros)