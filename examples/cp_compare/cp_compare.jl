import NTFk
import CanDecomp
import LinearAlgebra
import StaticArrays

quiet = false

Random.seed!(0)
A = rand(2, 3)
B = rand(5, 3)
C = rand(10, 3)
T = CanDecomp.totensor(A, B, C)

Af = rand(size(A)...);
Bf = rand(size(B)...);
Cf = rand(size(C)...);
@info("CanDecomp CP")
et = @elapsed CanDecomp.candecomp!(StaticArrays.SVector(Af, Bf, Cf), T, Val{:nnoptim}; regularization=1e-3, print_level=0, max_cd_iters=1000)
T_est = CanDecomp.totensor(Af, Bf, Cf);
@info("Norm $(LinearAlgebra.norm(T_est .- T))")
@warn("Execution time $et")
if !quiet
	NTFk.plot2matrices(A, Af)
	NTFk.plot2matrices(B, Bf)
	NTFk.plot2matrices(C, Cf)
	NTFk.plotlefttensor(T, T_est; progressbar=nothing)
end

@info("TensorDecompositions CP")
trank = 3
et = @elapsed t, csize, ibest = NTFk.analysis(T, [trank]; tol=1e-8, max_iter=1000);
T_est = TensorDecompositions.compose(t[ibest]);
@info("Norm $(LinearAlgebra.norm(T_est .- T))")
@warn("Execution time $et")
if !quiet
	NTFk.normalizefactors!(t[ibest])
	NTFk.plot2matrices(A, t[ibest].factors[1])
	NTFk.plot2matrices(B, t[ibest].factors[2])
	NTFk.plot2matrices(C, t[ibest].factors[3])
	NTFk.plotlefttensor(T, T_est; progressbar=nothing)
end

@info("MATLAB TensorToolBox cp_als")
et = @elapsed t = NTFk.ttanalysis(T, 3; maxiter=1000, tol=1e-8)
T_est = TensorDecompositions.compose(t);
@info("Norm $(LinearAlgebra.norm(T_est .- T))")
@warn("Execution time $et")
if !quiet
	NTFk.plot2matrices(A, t.factors[1])
	NTFk.plot2matrices(B, t.factors[2])
	NTFk.plot2matrices(C, t.factors[3])
	NTFk.plotlefttensor(T, t; progressbar=nothing)
end

@info("MATLAB TensorToolBox cp_npu")
et = @elapsed t = NTFk.ttanalysis(T, 3; maxiter=1000, tol=1e-8, functionname="cp_nmu")
T_est = TensorDecompositions.compose(t);
@info("Norm $(LinearAlgebra.norm(T_est .- T))")
@warn("Execution time $et")
if !quiet
	NTFk.plot2matrices(A, t.factors[1])
	NTFk.plot2matrices(B, t.factors[2])
	NTFk.plot2matrices(C, t.factors[3])
	NTFk.plotlefttensor(T, t; progressbar=nothing)
end

@info("MATLAB Block-Coordinate Update NCP")
et = @elapsed t = NTFk.bcuanalysis(T, 3; maxiter=1000, tol=1e-8, functionname="ncp")
T_est = TensorDecompositions.compose(t);
@info("Norm $(LinearAlgebra.norm(T_est .- T))")
if !quiet
	NTFk.plot2matrices(A, t.factors[1])
	NTFk.plot2matrices(B, t.factors[2])
	NTFk.plot2matrices(C, t.factors[3])
	NTFk.plotlefttensor(T, t; progressbar=nothing)
end

@info("TensorDecompositions Tucker with regularization")
t, csize, ibest = NTFk.analysis(T, [(3, 3, 3)]; eigmethod=[false,false,false], tol=1e-16, max_iter=1000)
T_est = TensorDecompositions.compose(t[ibest])
@info("Norm $(LinearAlgebra.norm(T_est .- T))")
@warn("Execution time $et")
if !quiet
	NTFk.normalizefactors!(t[ibest])
	NTFk.plot2matrices(A, t[ibest].factors[1])
	NTFk.plot2matrices(B, t[ibest].factors[2])
	NTFk.plot2matrices(C, t[ibest].factors[3])
	NTFk.plotlefttensor(T, T_est; progressbar=nothing)
end

@info("TensorDecompositions Tucker without regularization")
t, csize, ibest = NTFk.analysis(T, [(3, 3, 3)]; eigmethod=[false,false,false], lambda=0., tol=1e-16, max_iter=1000)
T_est = TensorDecompositions.compose(t[ibest])
@info("Norm $(LinearAlgebra.norm(T_est .- T))")
@warn("Execution time $et")
if !quiet
	NTFk.normalizefactors!(t[ibest])
	NTFk.plot2matrices(A, t[ibest].factors[1])
	NTFk.plot2matrices(B, t[ibest].factors[2])
	NTFk.plot2matrices(C, t[ibest].factors[3])
	NTFk.plotlefttensor(T, T_est; progressbar=nothing)
end