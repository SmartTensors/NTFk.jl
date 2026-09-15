import NTFk
import Test
import Random
import TensorDecompositions
import LinearAlgebra
import PyCall

# @Test.testset "NTFk ploting" begin
# 	r = reshape(repeat(collect(1/100:1/100:1), inner=100), (100,100))
# 	d = NTFk.plotmatrix(r; colormap=NTFk.colormap_hsv2);
# 	display(d); println()
# 	d = NTFk.plotmatrix(r; colormap=NTFk.colormap_hsv);
# 	display(d); println()
# 	@Test.test isapprox(1, 1, atol=1e-6)
# end

@Test.testset "NTFk TensorDecompositions Tucker analysis" begin
	Random.seed!(1)
	csize = (2, 3, 4)
	tsize = (5, 10, 15)
	tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
	T_orig = TensorDecompositions.compose(tucker_orig)
	tucker_est, csize_est, ibest = NTFk.analysis(T_orig, [csize], 1; seed=1, eigmethod=[false,false,false], progressbar=false, tol=1e-4, maxiter=100, lambda=0.)
	@Test.test csize == csize_est
	relative_residual::Float64 = TensorDecompositions.rel_residue(tucker_est[ibest], T_orig)
	@info "Relative reconstruction error: $(relative_residual)"
	@Test.test relative_residual < 0.03
end

@Test.testset "NTFk TensorDecompositions CP analysis" begin
	Random.seed!(1)
	csize = (2, 3, 4)
	tsize = (5, 10, 15)
	tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
	T_orig = TensorDecompositions.compose(tucker_orig)
	cp_est, csize_est, ibest = NTFk.analysis(T_orig, [maximum(csize)], 1; seed=1, eigmethod=[false,false,false], progressbar=false, tol=1e-12, maxiter=100, lambda=0.)
	@Test.test csize_est == maximum(csize)
	relative_residual::Float64 = TensorDecompositions.rel_residue(cp_est[ibest], T_orig)
	@info "Relative reconstruction error: $(relative_residual)"
	@Test.test relative_residual < 0.02
end

@Test.testset "NTFk TensorDecompositions Tucker sparse analysis" begin
	Random.seed!(1)
	csize = (2, 3, 4)
	tsize = (5, 10, 15)
	tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
	T_orig = TensorDecompositions.compose(tucker_orig)
	T_orig[1,1,:] .= NaN
	observed::BitArray{3} = .!isnan.(T_orig)
	tucker_est, csize_est, ibest = NTFk.analysis(T_orig, [csize], 1; seed=1, eigmethod=[false,false,false], progressbar=false, tol=1e-4, maxiter=100, lambda=0.)
	T_est = TensorDecompositions.compose(tucker_est[ibest])
	@Test.test csize == csize_est
	relative_residual::Float64 = LinearAlgebra.norm(T_orig[observed] .- T_est[observed]) / LinearAlgebra.norm(T_orig[observed])
	@info "Relative reconstruction error: $(relative_residual)"
	@Test.test relative_residual < 0.05
end

if NTFk.tensorly == PyCall.PyNULL()
	@warn("TensorLy is not available")
else
	@Test.testset "NTFk Tensorly Tucker analysis" begin
		Random.seed!(1)
		csize = (2, 3, 4)
		tsize = (5, 10, 15)
		tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
		T_orig = TensorDecompositions.compose(tucker_orig)
		for backend = ["pytorch", "mxnet", "numpy"]
			a = NTFk.analysis(T_orig, [csize], 1; seed=1, method="tensorly_", eigmethod=[false,false,false], progressbar=false, tol=1e-12, maxiter=100, backend=backend, verbose=true)
			if isnothing(a)
				continue
			end
			tucker_est, csize_est, ibest = a
			T_est = TensorDecompositions.compose(tucker_est[ibest])
			@Test.test csize == csize_est
			@info "Error $(LinearAlgebra.norm(T_orig .- T_est))"
			@Test.test isapprox(LinearAlgebra.norm(T_orig .- T_est), 0.913; atol=1e-3)
		end
	end
end

@Test.test 1 == 1
