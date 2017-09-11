import TensorDecompositions
import Combinatorics
include("helpers.jl")

srand(1)
tucker_orig = rand_tucker((2, 3, 4), (50, 100, 150), factors_nonneg=true, core_nonneg=true)
tnsr_orig = TensorDecompositions.compose(tucker_orig)
tnsr_max = maximum(tnsr_orig)
map!(x -> x / tnsr_max, tucker_orig.core, tucker_orig.core)
map!(x -> x / tnsr_max, tnsr_orig, tnsr_orig)

# tnsr = add_noise(tnsr_orig, 0.6, true)

tnsr = tnsr_orig

# Solve the problem
sizes = [(2,3,4), (1,3,4), (3,3,4), (2,2,4), (2,4,4), (2,3,3), (2,3,5)]
residues = Array{Float64}(0)
for s in sizes
	@time tucker_spnn = TensorDecompositions.spnntucker(tnsr, s, tol=1E-15, ini_decomp=:hosvd, core_nonneg=true, max_iter=1000, verbose=true, lambdas=fill(0.1, 4))
	push!(residues, TensorDecompositions.rel_residue(tucker_spnn))
end

# tnsr_est = TensorDecompositions.compose(tucker_spnn)

info("Relative error of decompositions:")
i = 1
for s in sizes
	println("$s --> $(residues[i])")
	i += 1
end