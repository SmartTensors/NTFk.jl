import NTFk

csize = (1, 1, 1)
tsize = (10, 20, 30)

xf = [x->x]
xfactor = Array{Float64}(undef, tsize[1], csize[1])
for i = 1:csize[1]
	x = linspace(0, 10, tsize[1])
	xfactor[:,i] = xf[i].(x)
end
xfactor = xfactor ./ maximum(xfactor, 1)
Mads.plotseries(xfactor)

yf = [y->y^2]
yfactor = Array{Float64}(undef, tsize[2], csize[2])
for i = 1:csize[2]
	y = linspace(0, 10, tsize[2])
	yfactor[:,i] = yf[i].(y)
end
yfactor = yfactor ./ maximum(yfactor, 1)
Mads.plotseries(yfactor)

zf = [z->exp(z)]
zfactor = Array{Float64}(undef, tsize[3], csize[3])
for i = 1:csize[3]
	z = linspace(0, 10, tsize[3])
	zfactor[:,i] = zf[i].(z)
end
zfactor = zfactor ./ maximum(zfactor, 1)
Mads.plotseries(zfactor)

core = zeros(csize)
core[1,1,1] = 1

# display(core)

tt_orig = TensorDecompositions.Tucker((xfactor, yfactor, zfactor), core)
T_orig = TensorDecompositions.compose(tt_orig)

# NTFk.plottensor(T_orig)

ttu, ecsize, ibest = NTFk.analysis(T_orig, [csize]; eigmethod=[false,false,false], lambda=0., tol=1e-16, max_iter=1000000, prefix="results/spnn-111")
NTFk.normalizecore!(ttu[ibest])
NTFk.normalizefactors!(ttu[ibest])
Mads.plotseries(xfactor)
Mads.plotseries(ttu[ibest].factors[1])
Mads.plotseries(yfactor)
Mads.plotseries(ttu[ibest].factors[2])
Mads.plotseries(zfactor)
Mads.plotseries(ttu[ibest].factors[3])

tcp, ecsize, ibest = NTFk.analysis(T_orig, [1]; tol=1e-16, max_iter=1000000, prefix="results/tdcp-111")
NTFk.normalizelambdas!(tcp[ibest])
NTFk.normalizefactors!(tcp[ibest])
Mads.plotseries(xfactor)
Mads.plotseries(tcp[ibest].factors[1])
Mads.plotseries(yfactor)
Mads.plotseries(tcp[ibest].factors[2])
Mads.plotseries(zfactor)
Mads.plotseries(tcp[ibest].factors[3])


