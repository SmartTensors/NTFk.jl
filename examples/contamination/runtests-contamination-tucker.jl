import NTFk
import TensorDecompositions
import TensorToolbox
import JLD

T_orig = JLD.load("tensor.jld", "T")
tsize = size(T_orig)

# T = add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [size(T) .- 1]
# sizes = [(1,2,2), (1,2,2), (1,2,2), (1,2,2), (1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3)]
ndimensons = length(sizes[1])
nruns = length(sizes)
residues = Array{Float64}(undef, nruns)
correlations = Array{Float64}(undef, nruns, ndimensons)
T_esta = Array{Array{Float64,3}}(undef, nruns)
tucker_spnn = Array{TensorDecompositions.Tucker{Float64,3}}(undef, nruns)
T_est = nothing
for i in 1:nruns
	@info("Core size: $(sizes[i])")
	@time tucker_spnn[i] = TensorDecompositions.spnntucker(T, sizes[i]; tol=1e-16, core_nonneg=true, verbose=false, max_iter=50000, lambdas=fill(0.1, length(sizes[i]) + 1))
	T_est = TensorDecompositions.compose(tucker_spnn[i])
	T_esta[i] = T_est
	residues[i] = TensorDecompositions.rel_residue(tucker_spnn[i])
	correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(T_est[:,k,j], T_orig[:,k,j]), 1:tsize[2])), 1:tsize[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(T_est[k,:,j], T_orig[k,:,j]), 1:tsize[1])), 1:tsize[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(T_est[k,j,:], T_orig[k,j,:]), 1:tsize[1])), 1:tsize[2]))
	println("$i - $(sizes[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core))")
end

ibest = 1
best = Inf
for i in 1:nruns
	if residues[i] < best
		best = residues[i]
		ibest = i
	end
end

csize = TensorToolbox.mrank(tucker_spnn[ibest].core)
NTFk.atensor(tucker_spnn[ibest].core)
ndimensons = length(csize)
@info("Estimated true core size: $(csize)")

sizes = [csize]
for i = 1:ndimensons
	push!(sizes, ntuple(k->(k == i ? csize[i] + 1 : csize[k]), ndimensons))
	push!(sizes, ntuple(k->(k == i ? csize[i] - 1 : csize[k]), ndimensons))
end

nruns = length(sizes)
residues = Array{Float64}(undef, nruns)
correlations = Array{Float64}(undef, nruns, ndimensons)
T_esta = Array{Array{Float64,3}}(undef, nruns)
tucker_spnn = Array{TensorDecompositions.Tucker{Float64,3}}(undef, nruns)
T_est = nothing
for i in 1:nruns
	@info("Core size: $(sizes[i])")
	@time tucker_spnn[i] = TensorDecompositions.spnntucker(T, sizes[i]; tol=1e-16, core_nonneg=true, verbose=false, max_iter=50000, lambdas=fill(0.1, length(sizes[i]) + 1))
	T_est = TensorDecompositions.compose(tucker_spnn[i])
	T_esta[i] = T_est
	residues[i] = TensorDecompositions.rel_residue(tucker_spnn[i])
	correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(T_est[:,k,j], T_orig[:,k,j]), 1:tsize[2])), 1:tsize[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(T_est[k,:,j], T_orig[k,:,j]), 1:tsize[1])), 1:tsize[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(T_est[k,j,:], T_orig[k,j,:]), 1:tsize[1])), 1:tsize[2]))
	println("$i - $(sizes[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core))")
end

@info("Decompositions:")
ibest = 1
best = Inf
for i in 1:nruns
	if residues[i] < best
		best = residues[i]
		ibest = i
	end
	println("$i - $(sizes[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) rank $(TensorToolbox.mrank(tucker_spnn[i].core))")
end

NTFk.plotcmptensors(T_orig, T_esta[ibest], 3)