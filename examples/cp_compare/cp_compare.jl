import NTFk
import CanDecomp

srand(0)
A = rand(3, 2)
B = rand(3, 50)
C = rand(3, 100)
T = CanDecomp.totensor(A, B, C)

Af = zeros(size(A))
Bf = zeros(size(B))
Cf = zeros(size(C))
info("CanDecomp CP")
@time CanDecomp.candecomp!(StaticArrays.SVector(Af, Bf, Cf), T; regularization=0, print_level=0, max_cd_iters=25)
T_est = CanDecomp.totensor(Af, Bf, Cf)
NTFk.plotcmptensors(T, T_est; progressbar=false)

info("TensorDecompositions CP")
trank = 3
@time t, csize, ibest = NTFk.analysis(T, [trank]; tol=1e-8, max_iter=1000)
T_est = TensorDecompositions.compose(t[ibest])
NTFk.plotcmptensors(T, T_est; progressbar=false)

info("TensorDecompositions Tucker with regularization")
t, csize, ibest = NTFk.analysis(T, [(2, 50, 100)])
T_est = TensorDecompositions.compose(t[ibest])
NTFk.plotcmptensors(T, T_est; progressbar=false)

info("MATLAB CanDecomp cp_als")
trank = 3
@time mt = NTFk.ttanalysis(T, trank)
NTFk.plotcmptensors(T, mt; progressbar=false)
