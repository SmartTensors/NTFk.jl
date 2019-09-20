import NTFk
import LinearAlgebra

csize = (3, 3, 3)
tsize = (20, 30, 40)

xf = [x->1., x->sin(x)+1, x->x^2]
xfactor = Array{Float64}(undef, tsize[1], csize[1])
for i = 1:csize[1]
	x = linspace(0, 10, tsize[1])
	xfactor[:,i] = xf[i].(x)
end
xfactor = xfactor ./ maximum(xfactor, 1)
xfactori = xfactor + rand(size(xfactor)) *.01
Mads.plotseries(xfactor)

yf = [y->1., y->cos(y), y->exp(y)]
yfactor = Array{Float64}(undef, tsize[2], csize[2])
for i = 1:csize[2]
	y = linspace(0, 10, tsize[2])
	yfactor[:,i] = yf[i].(y)
end
yfactor = yfactor ./ maximum(yfactor, 1)
yfactori = yfactor + rand(size(yfactor)) *.01
Mads.plotseries(yfactor)

zf = [z->1., z->cos(2z)+1, z->sqrt(z)]
zfactor = Array{Float64}(undef, tsize[3], csize[3])
for i = 1:csize[3]
	z = linspace(0, 10, tsize[3])
	zfactor[:,i] = zf[i].(z)
end
zfactor = zfactor ./ maximum(zfactor, 1)
zfactori = zfactor + rand(size(zfactor)) *.01
Mads.plotseries(zfactor)

core = zeros(csize)
core[2,1,1] = 1
core[3,2,2] = 1
core[1,3,1] = 1
core[1,1,3] = 1

# display(core)

tt_orig = TensorDecompositions.Tucker((xfactor, yfactor, zfactor), core)
tt_ini = TensorDecompositions.Tucker((xfactori, yfactori, zfactori), core)
T_orig = TensorDecompositions.compose(tt_orig)

# NTFk.plottensor(T_orig)

ths = TensorDecompositions.hosvd(T_orig, csize, [false,false,false]; pad_zeros=true, compute_error=true, compute_rank=false)
NTFk.normalizecore!(ths)
NTFk.normalizefactors!(ths)
Mads.plotseries(xfactor)
Mads.plotseries(ths.factors[1])
Mads.plotseries(yfactor)
Mads.plotseries(ths.factors[2])
Mads.plotseries(zfactor)
Mads.plotseries(ths.factors[3])

ttu, ecsize, ibest = NTFk.analysis(T_orig, [csize], 1; eigmethod=[false,false,false], max_iter=100000, lambda=0., prefix="results/spnn-333")
# ttu, ecsize, ibest = NTFk.analysis(T_orig, [(3,30,40)], 1; eigmethod=[false,false,false], lambda=1., prefix="results/spnn-33040")
T_est = TensorDecompositions.compose(ttu[ibest]);
@info("Norm $(LinearAlgebra.norm(T_orig .- T_est))")
NTFk.normalizecore!(ttu[ibest])
NTFk.normalizefactors!(ttu[ibest])
Mads.plotseries(xfactor)
Mads.plotseries(ttu[ibest].factors[1])
Mads.plotseries(yfactor)
Mads.plotseries(ttu[ibest].factors[2])
Mads.plotseries(zfactor)
Mads.plotseries(ttu[ibest].factors[3])

tcp, ecsize, ibest = NTFk.analysis(T_orig, [3]; prefix="results/tdcp-333")
NTFk.normalizelambdas!(tcp[ibest])
NTFk.normalizefactors!(tcp[ibest])
Mads.plotseries(xfactor)
Mads.plotseries(tcp[ibest].factors[1])
Mads.plotseries(yfactor)
Mads.plotseries(tcp[ibest].factors[2])
Mads.plotseries(zfactor)
Mads.plotseries(tcp[ibest].factors[3])


