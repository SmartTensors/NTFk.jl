import dNTF

srand(0)
A = rand(3, 2)
B = rand(3, 50)
C = rand(3, 100)
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

info("TensorDecompositions Tucker with regularization")
trank = 2
t, csize = dNTF.analysis(T, [(3, 50, 100)]; tol=1e-8, ini_decomp=:hosvd, core_nonneg=true, verbose=true, max_iter=1000, lambda=0.1)
nt = TensorDecompositions.compose(t[1])
dNTF.plotcmptensor(T, nt; progressbar=false)

info("MATLAB CanDecomp cp_als")
trank = 3
@time mt = dNTF.manalysis(T, trank)
dNTF.plotcmptensor(T, mt; progressbar=false)
