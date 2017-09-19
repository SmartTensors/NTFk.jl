import TensorDecompositions
import Combinatorics

size_tnst = (2, 10, 5)
tnsr_orig = Array{Float64}(size_tnst)
tnsr_orig[:,:,1] = [ones(2,2) zeros(2,8)]
tnsr_orig[:,:,2] = [zeros(2,2) ones(2,2) zeros(2,6)]
tnsr_orig[:,:,3] = [zeros(2,4) ones(2,2) zeros(2,4)]
tnsr_orig[:,:,4] = [zeros(2,6) ones(2,2) zeros(2,2)]
tnsr_orig[:,:,5] = [zeros(2,8) ones(2,2)]
tnsr_orig .*= 100

# tnsr = add_noise(tnsr_orig, 0.6, true)
tnsr = tnsr_orig

sizes = [(1,4,4), (1,4,4), (1,4,4), (1,4,4), (1,5,4), (1,5,4), (1,5,4), (1,5,4), (1,5,4)]
# sizes = [(1,2,2), (1,2,2), (1,2,2), (1,2,2), (1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3)]
ndimensons = length(sizes[1])
nruns = length(sizes)
residues = Array{Float64}(nruns)
correlations_factors = Array{Float64}(nruns, ndimensons)
correlations = Array{Float64}(nruns, ndimensons)
tnsr_esta = Array{Array{Float64,3}}(nruns)
tucker_spnn = Array{TensorDecompositions.Tucker{Float64,3}}(nruns)
for i in 1:nruns
	@time tucker_spnn[i] = TensorDecompositions.spnntucker(tnsr, sizes[i], tol=1e-16, ini_decomp=:hosvd, core_nonneg=true, max_iter=1000, verbose=false, lambdas=fill(0.1, 4))
	tnsr_est = TensorDecompositions.compose(tucker_spnn[i])
	tnsr_esta[i] = tnsr_est
	residues[i] = TensorDecompositions.rel_residue(tucker_spnn[i])
	@show sizes[i]
	correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(tnsr_est[:,k,j], tnsr_orig[:,k,j]), 1:size_tnst[2])), 1:size_tnst[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(tnsr_est[k,:,j], tnsr_orig[k,:,j]), 1:size_tnst[1])), 1:size_tnst[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(tnsr_est[k,j,:], tnsr_orig[k,j,:]), 1:size_tnst[1])), 1:size_tnst[2]))
end

info("Relative error of decompositions:")
ibest = 1
best = Inf
for i in 1:nruns
	if residues[ibest] < best
		ibest = i
	end
	println("$i - $(sizes[i]): residual $(residues[i]) correlations $(correlations[i,:])")
end

plotcmptensor(tnsr_orig, tnsr_esta[ibest], 3)