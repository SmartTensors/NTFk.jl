import dNTF
import TensorDecompositions
import JLD

T_orig = JLD.load("tensor.jld", "T")

# T = add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [size(T)]
# sizes = [(1,2,2), (1,2,2), (1,2,2), (1,2,2), (1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3)]
ndimensons = length(sizes[1])
nruns = length(sizes)
residues = Array{Float64}(nruns)
correlations = Array{Float64}(nruns, ndimensons)
T_esta = Array{Array{Float64,3}}(nruns)
tucker_spnn = Array{TensorDecompositions.Tucker{Float64,3}}(nruns)
for i in 1:nruns
	info("Core size: $(sizes[i])")
	@time tucker_spnn[i] = TensorDecompositions.spnntucker(T, sizes[i], tol=1e-16, ini_decomp=:hosvd, core_nonneg=true, max_iter=100000, verbose=false, lambdas=fill(0.1, length(sizes[i]) + 1))
	T_est = TensorDecompositions.compose(tucker_spnn[i])
	T_esta[i] = T_est
	residues[i] = TensorDecompositions.rel_residue(tucker_spnn[i])
end

info("Relative error of decompositions:")
ibest = 1
best = Inf
for i in 1:nruns
	if residues[i] < best
		best = residues[i]
		ibest = i
	end
	println("$i - $(sizes[i]): residual $(residues[i]) tensor correlations $(correlations[i,:])")
end

dNTF.plotcmptensor(T_orig, T_esta[ibest], 3)