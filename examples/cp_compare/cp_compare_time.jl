import NTFk

Random.seed!(0)
A = rand(2, 3)
B = rand(50, 3)
C = rand(100, 3)
T = CanDecomp.totensor(A, B, C)

Af = rand(size(A));
Bf = rand(size(B));
Cf = rand(size(C));
@info("CanDecomp CP Jump")
@time CanDecomp.candecomp!(StaticArrays.SVector(Af, Bf, Cf), T, Val{:nnjump}; regularization=0, print_level=0, max_cd_iters=25)

Af = rand(size(A));
Bf = rand(size(B));
Cf = rand(size(C));
@info("CanDecomp CP Optim")
@time CanDecomp.candecomp!(StaticArrays.SVector(Af, Bf, Cf), T, Val{:nnoptim}; regularization=0, print_level=0, max_cd_iters=25)

Af = rand(size(A));
Bf = rand(size(B));
Cf = rand(size(C));
@info("CanDecomp CP Mads")
@time CanDecomp.candecomp!(StaticArrays.SVector(Af, Bf, Cf), T, Val{:nnmads}; regularization=0, print_level=0, max_cd_iters=25)

@info("TensorDecompositions CP")
trank = 3
@time t, csize, ibest = NTFk.analysis(T, [trank]; tol=1e-8, max_iter=1000)

@info("TensorDecompositions Tucker with regularization")
t, csize, ibest = NTFk.analysis(T, [(3, 3, 3)])

@info("MATLAB TensorToolBox CanDecomp cp_als")
trank = 3
@time mt = NTFk.ttanalysis(T, trank)
