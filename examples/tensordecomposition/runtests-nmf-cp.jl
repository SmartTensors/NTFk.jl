import NTFk

Random.seed!(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
Q = [1; 2; 3]
X = W * H
tsize = (20, 5, 3)
T_orig = Array{Float64}(undef, tsize)
T_orig[:,:,1] = X
T_orig[:,:,2] = X * 2
T_orig[:,:,3] = X * 3

# T = NTFk.add_noise(T_orig, 0.6, true)
T = T_orig

tranks = [2, 3, 4]
cpf, csize, ibest = NTFk.analysis(T, tranks, 10; method=:cp_als, quiet=false)

NTFk.plotcmptensors(T_orig, T_esta[ibest], 3; progressbar=false)
@show ibest
@show cor(W[:,1], cpf[ibest].factors[1][:,1])
@show cor(W[:,2], cpf[ibest].factors[1][:,2])
@show cor(H[1,:], cpf[ibest].factors[2][:,1])
@show cor(H[2,:], cpf[ibest].factors[2][:,2])
@show cor(Q, cpf[ibest].factors[3][:,1])
:done