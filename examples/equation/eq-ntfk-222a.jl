import NTFk

csize = (2, 2, 2)
tsize = (20, 20, 20)

xf = [x->sin(x)+1, x->x^2]
xfactor = Array{Float64}(undef, tsize[1], csize[1])
for i = 1:csize[1]
	x = linspace(0, 10, tsize[1])
	xfactor[:,i] = xf[i].(x)
end
xfactor = xfactor ./ maximum(xfactor, 1)
Mads.plotseries(xfactor)

yf = [y->1, y->cos(y)+1]
yfactor = Array{Float64}(undef, tsize[2], csize[2])
for i = 1:csize[2]
	y = linspace(0, 10, tsize[2])
	yfactor[:,i] = yf[i].(y)
end
# yfactor = yfactor ./ maximum(yfactor, 1)
Mads.plotseries(yfactor)

zf = [z->1, z->cos(z)+1]
zfactor = Array{Float64}(undef, tsize[3], csize[3])
for i = 1:csize[3]
	z = linspace(0, 10, tsize[3])
	zfactor[:,i] = zf[i].(z)
end
# zfactor = zfactor ./ maximum(zfactor, 1)
Mads.plotseries(zfactor)

core = zeros(csize)
core[1,1,1] = 1
core[2,2,2] = 1

# display(core)

tt_orig = TensorDecompositions.Tucker((xfactor, yfactor, zfactor), core)
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

ttu, ecsize, ibest = NTFk.analysis(T_orig, [csize], 1; eigmethod=[false,false,false], max_iter=100000, lambda=0., prefix="results/spnn-222a")
NTFk.normalizecore!(ttu[ibest])
NTFk.normalizefactors!(ttu[ibest])
Mads.plotseries(xfactor)
Mads.plotseries(ttu[ibest].factors[1])
Mads.plotseries(yfactor)
Mads.plotseries(ttu[ibest].factors[2])
Mads.plotseries(zfactor)
Mads.plotseries(ttu[ibest].factors[3])

Wem, Hem, of, rob, aic = NMFk.execute(NTFk.flatten(T_orig, 1)', 2:2)
NTFk.plot2d(Wem[2]')

tcp, ecsize, ibest = NTFk.analysis(T_orig, [2]; prefix="results/tdcp-222a")
NTFk.normalizelambdas!(tcp[ibest])
NTFk.normalizefactors!(tcp[ibest])
Mads.plotseries(xfactor)
Mads.plotseries(tcp[ibest].factors[1])
Mads.plotseries(yfactor)
Mads.plotseries(tcp[ibest].factors[2])
Mads.plotseries(zfactor)
Mads.plotseries(tcp[ibest].factors[3])


