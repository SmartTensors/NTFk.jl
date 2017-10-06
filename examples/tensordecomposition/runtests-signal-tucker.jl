import TensorDecompositions
import Combinatorics
dntfdir = splitdir(Base.source_path())[1]
include(joinpath(dntfdir, "helpers.jl"))
include(joinpath(dntfdir, "..", "..", "src", "display.jl"))

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

sizes = [(1,4,4), (1,4,4), (1,4,4), (1,4,4), (1,5,4), (1,5,4), (1,5,4), (1,5,4), (1,5,4)]
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
	println("$i - $(sizes[i]): residual $(residues[i]) tensor correlations $(correlations[i,:])")
end

plotcmptensor(T_orig, T_esta[ibest], 3)