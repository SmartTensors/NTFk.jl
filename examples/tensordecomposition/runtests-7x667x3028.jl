import TensorDecompositions
import dNTF

srand(1)
size_core_orig = (2, 3, 4)
tsize = (7, 667, 3038)
# tsize = (5, 10, 15)
tucker_orig = dNTF.rand_tucker(size_core_orig, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(tucker_orig)
T_max = maximum(T_orig)
map!(x -> x / T_max, tucker_orig.core, tucker_orig.core)
map!(x -> x / T_max, T_orig, T_orig)

# T = add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [size_core_orig, (1,3,4), (3,3,4), (2,2,4), (2,4,4), (2,3,3), (2,3,5)]
sizes = [size_core_orig]
ndimensons = length(size_core_orig)
nruns = length(sizes)
residues = Array{Float64}(nruns)
correlations_factors = Array{Float64}(nruns, ndimensons)
correlations = Array{Float64}(nruns, ndimensons)
T_est = 0
tucker_spnn = 0
for i in 1:nruns
	info("Core size: $(sizes[i])")
	@time tucker_spnn = TensorDecompositions.spnntucker(T, sizes[i], tol=1e-15, ini_decomp=:hosvd, core_nonneg=true, max_iter=1000, verbose=true, lambdas=fill(0.1, 4))
	T_est = TensorDecompositions.compose(tucker_spnn)
	residues[i] = TensorDecompositions.rel_residue(tucker_spnn)
	for j = 1:ndimensons
		correlations_factors[i,j] = minimum(map(k->cor(tucker_orig.factors[j][:,k], tucker_spnn.factors[j][:,k]), 1:min(sizes[i][j],size_core_orig[j])))
	end
	correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(T_est[:,k,j], T_orig[:,k,j]), 1:tsize[2])), 1:tsize[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(T_est[k,:,j], T_orig[k,:,j]), 1:tsize[1])), 1:tsize[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(T_est[k,j,:], T_orig[k,j,:]), 1:tsize[1])), 1:tsize[2]))
end

info("Relative error of decompositions:")
for i in 1:nruns
	println("$(sizes[i]): residual $(residues[i]) correlations $(correlations[i,:]) correlations_factors $(correlations_factors[i,:])")
end