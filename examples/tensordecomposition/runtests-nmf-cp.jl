import TensorDecompositions
import dNTF

srand(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
Q = [1; 2; 3]
X = W * H
tsize = (20, 5, 3)
T_orig = Array{Float64}(tsize)
T_orig[:,:,1] = X
T_orig[:,:,2] = X * 2
T_orig[:,:,3] = X * 3

# T = add_noise(T_orig, 0.6, true)
T = T_orig

tranks = [1, 2, 3, 4]
ndimensons = length(size(T))
nruns = length(tranks)
residues = Array{Float64}(nruns)
correlations_factors = Array{Float64}(nruns, ndimensons)
correlations = Array{Float64}(nruns, ndimensons)
T_esta = Array{Array{Float64,3}}(nruns)
cpf = Array{TensorDecompositions.CANDECOMP{Float64,3}}(nruns)
for i in 1:nruns
	factors_initial_guess = tuple([randn(dim, tranks[i]) for dim in tsize]...)
	@time cpf[i] = TensorDecompositions.candecomp(T, tranks[i], factors_initial_guess, compute_error=true, method=:ALS, maxiter=1000)
	T_est = TensorDecompositions.compose(cpf[i])
	T_esta[i] = T_est
	residues[i] = TensorDecompositions.rel_residue(cpf[i])
	correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(T_est[:,k,j], T_orig[:,k,j]), 1:tsize[2])), 1:tsize[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(T_est[k,:,j], T_orig[k,:,j]), 1:tsize[1])), 1:tsize[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(T_est[k,j,:], T_orig[k,j,:]), 1:tsize[1])), 1:tsize[2]))
	println("$(tranks[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) factor correlations $(correlations_factors[i,:])")
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
@show cor(W[:,1], cpf[ibest].factors[1][:,1])
@show cor(W[:,2], cpf[ibest].factors[1][:,2])
@show cor(H[1,:], cpf[ibest].factors[2][:,1])
@show cor(H[2,:], cpf[ibest].factors[2][:,2])
@show cor(Q, cpf[ibest].factors[3][:,1])