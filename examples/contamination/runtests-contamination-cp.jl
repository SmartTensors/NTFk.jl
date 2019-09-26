import NTFk
import TensorDecompositions
import JLD
import Statistics

T_orig = JLD.load("tensor.jld", "T")
tsize = size(T_orig)

# T = add_noise(T_orig, 0.6, true)
T = T_orig

tranks = collect(1:minimum(tsize))
# sizes = [(1,2,2), (1,2,2), (1,2,2), (1,2,2), (1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3)]
ndimensons = length(tsize)
nruns = length(tranks)
residues = Array{Float64}(undef, nruns)
correlations = Array{Float64}(undef, nruns, ndimensons)
T_esta = Array{Array{Float64,3}}(undef, nruns)
cpf = Array{TensorDecompositions.CANDECOMP{Float64,3}}(undef, nruns)
for i in 1:nruns
	@info("CP rank: $(tranks[i])")
	factors_initial_guess = tuple([randn(dim, tranks[i]) for dim in tsize]...)
	@time cpf[i] = TensorDecompositions.candecomp(T, tranks[i], factors_initial_guess, compute_error=true, method=:ALS)
	T_est = TensorDecompositions.compose(cpf[i])
	T_esta[i] = T_est
	residues[i] = TensorDecompositions.rel_residue(cpf[i])
	correlations[i,1] = minimum(map((j)->minimum(map((k)->Statistics.cor(T_est[:,k,j], T_orig[:,k,j]), 1:tsize[2])), 1:tsize[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->Statistics.cor(T_est[k,:,j], T_orig[k,:,j]), 1:tsize[1])), 1:tsize[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->Statistics.cor(T_est[k,j,:], T_orig[k,j,:]), 1:tsize[1])), 1:tsize[2]))
	println("$i - $(tranks[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) ")
end

@info("Decompositions:")
ibest = 1
best = Inf
for i in 1:nruns
	if residues[i] < best
		best = residues[i]
		ibest = i
	end
	println("$i - $(tranks[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) ")
end

NTFk.plotcmptensors(T_orig, T_esta[ibest], 3)