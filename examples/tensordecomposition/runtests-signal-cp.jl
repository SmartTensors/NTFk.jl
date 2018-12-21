import NTFk

tsize = (2, 10, 5)
T_orig = Array{Float64}(undef, tsize)
T_orig[:,:,1] = [ones(2,2) zeros(2,8)]
T_orig[:,:,2] = [zeros(2,2) ones(2,2) zeros(2,6)]
T_orig[:,:,3] = [zeros(2,4) ones(2,2) zeros(2,4)]
T_orig[:,:,4] = [zeros(2,6) ones(2,2) zeros(2,2)]
T_orig[:,:,5] = [zeros(2,8) ones(2,2)]
T_orig .*= 100

# T = NTFk.add_noise(T_orig, 0.6, true)
T = T_orig

tranks = [1, 2, 3, 4, 5]
t, c, ibest = NTFk.analysis(T, tranks, 10; method=:cp_nmu)
NTFk.plotcmptensors(T, t[ibest], 3; progressbar=false)