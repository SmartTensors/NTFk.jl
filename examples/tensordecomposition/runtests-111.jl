import TensorDecompositions
import Combinatorics
include("helpers.jl")

srand(1)
tucker_orig = rand_tucker((1, 1, 1), (5, 10, 15), factors_nonneg=true, core_nonneg=true)
tnsr_orig = TensorDecompositions.compose(tucker_orig)
tnsr_max = maximum(tnsr_orig)
map!(x -> x / tnsr_max, tucker_orig.core, tucker_orig.core)
map!(x -> x / tnsr_max, tnsr_orig, tnsr_orig)

# tnsr = add_noise(tnsr_orig, 0.6, true)

tnsr = tnsr_orig

# Solve the problem
sizes = [(1,1,1)]
ndimensons = length(sizes[1])
nruns = length(sizes)
residues = Array{Float64}(nruns)
correlations_factors = Array{Float64}(nruns, ndimensons)
correlations = Array{Float64}(nruns, ndimensons)
tnsr_est = 0
tucker_spnn = 0
for i in 1:nruns
	@time tucker_spnn = TensorDecompositions.spnntucker(tnsr, sizes[i], tol=1e-15, ini_decomp=:hosvd, core_nonneg=true, max_iter=1000, verbose=true, lambdas=fill(0.1, 4))
	tnsr_est = TensorDecompositions.compose(tucker_spnn)
	residues[i] = TensorDecompositions.rel_residue(tucker_spnn)
	@show sizes[i]
	for j = 1:ndimensons
		correlations_factors[i,j] = minimum(map(k->cor(tucker_orig.factors[j][:,k], tucker_spnn.factors[j][:,k]), 1:min(sizes[i][j],size_core_orig[j])))
	end
	correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(tnsr_est[:,k,j], tnsr_orig[:,k,j]), 1:size_tnst[2])), 1:size_tnst[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(tnsr_est[k,:,j], tnsr_orig[k,:,j]), 1:size_tnst[1])), 1:size_tnst[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(tnsr_est[k,j,:], tnsr_orig[k,j,:]), 1:size_tnst[1])), 1:size_tnst[2]))
end

info("Relative error of decompositions:")
i = 1
for s in sizes
	println("$s: residual $(residues[i]) correlations $(correlations[i,:]) correlations_factors $(correlations_factors[i,:])")
	i += 1
end