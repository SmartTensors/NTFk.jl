import NTFk

srand(0)
A = rand(2, 3)
B = rand(50, 3)
C = rand(100, 3)
T = CanDecomp.totensor(A, B, C)

Af = zeros(size(A))
Bf = zeros(size(B))
Cf = zeros(size(C))
info("CanDecomp CP")
@time CanDecomp.candecomp!(StaticArrays.SVector(Af, Bf, Cf), T; regularization=0, print_level=0, max_cd_iters=25)
T_est = CanDecomp.totensor(Af, Bf, Cf)
NTFk.plotcmptensors(T, T_est; progressbar=nothing)

info("TensorDecompositions CP")
trank = 3
@time t, csize, ibest = NTFk.analysis(T, [trank]; tol=1e-8, max_iter=1000)
T_est = TensorDecompositions.compose(t[ibest])
NTFk.plotlefttensor(T, T_est; progressbar=nothing)

info("TensorDecompositions Tucker with regularization")
t, csize, ibest = NTFk.analysis(T, [(3, 3, 3)])
T_est = TensorDecompositions.compose(t[ibest])
NTFk.plotlefttensor(T, T_est; progressbar=nothing)

info("MATLAB TensorToolBox CanDecomp cp_als")
trank = 3
@time mt = NTFk.ttanalysis(T, trank)
NTFk.plotlefttensor(T, mt; progressbar=nothing)
