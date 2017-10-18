import TensorDecompositions
import dNTF

tsize = (2, 10, 5)
T_orig = Array{Float64}(tsize)
T_orig[:,:,1] = [ones(2,2) zeros(2,8)]
T_orig[:,:,2] = [zeros(2,2) ones(2,2) zeros(2,6)]
T_orig[:,:,3] = [zeros(2,4) ones(2,2) zeros(2,4)]
T_orig[:,:,4] = [zeros(2,6) ones(2,2) zeros(2,2)]
T_orig[:,:,5] = [zeros(2,8) ones(2,2)]
T_orig .*= 100

# T = add_noise(T_orig, 0.6, true)
T = T_orig

tranks = [1, 2]
ndimensons = length(size(T))
nruns = length(tranks)
residues = Array{Float64}(nruns)
correlations_factors = Array{Float64}(nruns, ndimensons)
correlations = Array{Float64}(nruns, ndimensons)
T_esta = Array{Array{Float64,3}}(nruns)
cpf = Array{TensorDecompositions.CANDECOMP{Float64,3}}(nruns)
for i in 1:nruns
	info("CP rank: $(tranks[i])")
	factors_initial_guess = tuple([randn(dim, tranks[i]) for dim in tsize]...)
	@time cpf[i] = TensorDecompositions.candecomp(T, tranks[i], factors_initial_guess, compute_error=true, method=:ALS)
	T_est = TensorDecompositions.compose(cpf[i])
	T_esta[i] = T_est
	residues[i] = TensorDecompositions.rel_residue(cpf[i])
	correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(T_est[:,k,j], T_orig[:,k,j]), 1:tsize[2])), 1:tsize[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(T_est[k,:,j], T_orig[k,:,j]), 1:tsize[1])), 1:tsize[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(T_est[k,j,:], T_orig[k,j,:]), 1:tsize[1])), 1:tsize[2]))
end

info("Relative error of decompositions:")
ibest = 1
best = Inf
for i in 1:nruns
	if residues[i] < best
		best = residues[i]
		ibest = i
	end
	println("$(tranks[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) factor correlations $(correlations_factors[i,:])")
end

dNTF.plotcmptensor(T_orig, T_esta[ibest], 3)