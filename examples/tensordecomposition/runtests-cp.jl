import TensorDecompositions
import Combinatorics
dntfdir = splitdir(Base.source_path())[1]
include(joinpath(dntfdir, "helpers.jl"))
include(joinpath(dntfdir, "..", "..", "src", "display.jl"))

srand(1)
trank = 2
tsize = (10, 20, 5)
factors_orig = rand_candecomp(trank, tsize, lambdas_nonneg=true, factors_nonneg=true)
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
	residues[i] = TensorDecompositions.rel_residue(factors)
	for j = 1:ndimensons
		correlations_factors[i,j] = minimum(map(k->cor(factors_orig.factors[j][:,k], factors.factors[j][:,k]), 1:min(size(factors.factors[j], 2), size(factors_orig.factors[j], 2))))
	end
	correlations[i,1] = minimum(map((j)->minimum(map((k)->cor(T_est[:,k,j], T_orig[:,k,j]), 1:tsize[2])), 1:tsize[3]))
	correlations[i,2] = minimum(map((j)->minimum(map((k)->cor(T_est[k,:,j], T_orig[k,:,j]), 1:tsize[1])), 1:tsize[3]))
	correlations[i,3] = minimum(map((j)->minimum(map((k)->cor(T_est[k,j,:], T_orig[k,j,:]), 1:tsize[1])), 1:tsize[2]))
end

info("Relative error of decompositions:")
for i in 1:nruns
	println("$(tranks[i]): residual $(residues[i]) tensor correlations $(correlations[i,:]) factor correlations $(correlations_factors[i,:])")
end