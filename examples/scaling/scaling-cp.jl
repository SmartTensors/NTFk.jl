import TensorDecompositions
import dNTF

srand(1)
trank = 2
tsize = (10, 20, 5)
factors_orig = dNTF.rand_candecomp(trank, tsize, lambdas_nonneg=true, factors_nonneg=true)
T_orig = TensorDecompositions.compose(factors_orig)

# T = add_noise(T_orig, 0.6, true)
T = T_orig

tranks = [1, 2, 3]
ndimensons = length(size(T))
nruns = length(tranks)
residues = Array{Float64}(nruns)
correlations_factors = Array{Float64}(nruns, ndimensons)
correlations = Array{Float64}(nruns, ndimensons)
T_est = 0
for i in 1:nruns
	factors_initial_guess = tuple([randn(dim, tranks[i]) for dim in tsize]...)
	@time factors = TensorDecompositions.candecomp(T, tranks[i], factors_initial_guess, compute_error=true, method=:ALS)
	T_est = TensorDecompositions.compose(factors)
end
