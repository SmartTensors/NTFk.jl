import dNTF

srand(0)
A = rand(3, 2)
B = rand(3, 5)
C = rand(3, 10)
T = CanDecomp.totensor(A, B, C)

Af = zeros(size(A))
Bf = zeros(size(B))
Cf = zeros(size(C))
@time CanDecomp.candecomp!(StaticArrays.SVector(Af, Bf, Cf), T; regularization=0, print_level=0, max_cd_iters=25)
T_est = CanDecomp.totensor(Af, Bf, Cf)
dNTF.plotcmptensor(T, T_est; progressbar=false)

info("TensorDecompositions CP")
trank = 3
@time t, csize = dNTF.analysis(T, [trank]; tol=1e-8, verbose=true, max_iter=1000)
nt = TensorDecompositions.compose(t[1])
dNTF.plotcmptensor(T, nt; progressbar=false)

info("MATLAB CanDecomp cp_als")
trank = 3
@time mt = dNTF.manalysis(T, trank)
dNTF.plotcmptensor(T, mt; progressbar=false)
