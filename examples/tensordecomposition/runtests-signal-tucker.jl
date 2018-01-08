import dNTF

tsize = (2, 10, 7)
T_orig = Array{Float64}(tsize)
T_orig[:,:,1] = zeros(2,10)
T_orig[:,:,2] = [ones(2,2) zeros(2,8)]
T_orig[:,:,3] = [zeros(2,2) ones(2,2) zeros(2,6)]
T_orig[:,:,4] = [zeros(2,4) ones(2,2) zeros(2,4)]
T_orig[:,:,5] = [zeros(2,6) ones(2,2) zeros(2,2)]
T_orig[:,:,6] = [zeros(2,8) ones(2,2)]
T_orig[:,:,7] = zeros(2,10)
T_orig .*= 100

# T = dNTF.add_noise(T_orig, 0.6, true)
T = T_orig

sizes = [(1,4,4), (1,4,4), (1,4,4), (1,4,4), (1,5,4), (1,5,4), (1,5,4), (1,5,4), (1,5,4)]
# sizes = [(1,2,2), (1,2,2), (1,2,2), (1,2,2), (1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3)]
t, c, ibest = dNTF.analysis(T, [tsize]; progressbar=true, tol=1e-16, max_iter=50000)
dNTF.plotcmptensor(T, t[ibest], 3; progressbar=false)
dNTF.plotmatrix(t[ibest].factors[1])
dNTF.plotmatrix(t[ibest].factors[2])
dNTF.plotmatrix(t[ibest].factors[3])
dNTF.plotmatrix(t[ibest].core[1,:,:])

map!(x -> x / 100, t[ibest].core, t[ibest].core)