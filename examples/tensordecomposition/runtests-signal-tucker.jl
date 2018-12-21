import NTFk

tsize = (2, 10, 7)
T_orig = Array{Float64}(undef, tsize)
T_orig[:,:,1] = zeros(2,10)
T_orig[:,:,2] = [ones(2,2) zeros(2,8)]
T_orig[:,:,3] = [zeros(2,2) ones(2,2) zeros(2,6)]
T_orig[:,:,4] = [zeros(2,4) ones(2,2) zeros(2,4)]
T_orig[:,:,5] = [zeros(2,6) ones(2,2) zeros(2,2)]
T_orig[:,:,6] = [zeros(2,8) ones(2,2)]
T_orig[:,:,7] = zeros(2,10)
T_orig .*= 100

# T = NTFk.add_noise(T_orig, 0.6, true)
T = T_orig

t, c, ibest = NTFk.analysis(T, [tsize], 5; progressbar=true, tol=1e-16, max_iter=10000)
NTFk.plotcmptensors(T, t[ibest], 3; progressbar=false)
NTFk.plotmatrix(t[ibest].factors[1])
NTFk.plotmatrix(t[ibest].factors[2])
NTFk.plotmatrix(t[ibest].factors[3])
NTFk.plotmatrix(t[ibest].core[1,:,:])