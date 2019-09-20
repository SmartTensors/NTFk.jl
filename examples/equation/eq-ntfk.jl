import NTFk

csize = (3, 4, 5)
tsize = (20, 50, 50)

xf = [x->x, x->x^2, x->1+tanh(x-5.2)]
xfactor = Array{Float64}(undef, tsize[1], csize[1]);
for i = 1:csize[1]
	x = linspace(0, 10, tsize[1])
	xfactor[:,i] = xf[i].(x)
end
xfactor = xfactor ./ maximum(xfactor, 1);
Mads.plotseries(xfactor, "figures-results/spnn-345-xfactors-true.png"; xaxis=0:19)

yf = [y->y, y->y^3, y->exp(y), y->sin(y)+1]
yfactor = Array{Float64}(undef, tsize[2], csize[2]);
for i = 1:csize[2]
	y = linspace(0, 10, tsize[2])
	yfactor[:,i] = yf[i].(y)
end
yfactor = yfactor ./ maximum(yfactor, 1);
Mads.plotseries(yfactor, "figures-results/spnn-345-yfactors-true.png"; xaxis=0:49)

zf = [z->z, z->z^4, z->log(z+1), z->sin(2z)+1, z->cos(z)+1]
zfactor = Array{Float64}(undef, tsize[3], csize[3]);
for i = 1:csize[3]
	z = linspace(0, 10, tsize[3])
	zfactor[:,i] = zf[i].(z)
end
zfactor = zfactor ./ maximum(zfactor, 1);
Mads.plotseries(zfactor, "figures-results/spnn-345-zfactors-true.png"; xaxis=0:49)

core = zeros(csize)
core[1,1,1] = 1
core[1,1,2] = 1
core[1,1,3] = 1
core[1,1,4] = 1
core[1,1,5] = 1

# core[1,2,1] = 1/2
# core[2,3,2] = 1/2

# core[1,1,3] = 1/3
# core[2,2,4] = 1/3

core[1,2,1] = 1
core[1,3,1] = 1
core[1,4,1] = 1

core[2,1,1] = 1
core[3,1,1] = 1

core[2,3,4] = 1
core[3,4,5] = 1

# display(core)

tt_orig = TensorDecompositions.Tucker((xfactor, yfactor, zfactor), core)
T_orig = TensorDecompositions.compose(tt_orig)

NTFk.plottensor(T_orig; maxvalue=2, movie=true, prefix="figures-results/spnn-345")
NTFk.plot2d(NTFk.flatten(T_orig, 1)')
Mads.plotseries(NTFk.flatten(T_orig, 1))

xfactori = xfactor + rand(size(xfactor)) *.01
yfactori = yfactor + rand(size(yfactor)) *.01
zfactori = zfactor + rand(size(zfactor)) *.01
tt_ini = TensorDecompositions.Tucker((xfactori, yfactori, zfactori), core)

ttu, ecsize, ibest = NTFk.analysis(T_orig, [csize]; eigmethod=[false,false,false], lambda=0., max_iter=10000, ini_decomp=tt_ini, prefix="results/spnn-345")
T_est = TensorDecompositions.compose(ttu[ibest]);
@info("Norm $(LinearAlgebra.norm(T_orig .- T_est))")
NTFk.normalizecore!(ttu[ibest])
NTFk.normalizefactors!(ttu[ibest])
# NTFk.plot2matrices(xfactor, ttu[ibest].factors[1])
# NTFk.plot2matrices(yfactor, ttu[ibest].factors[2])
# NTFk.plot2matrices(zfactor, ttu[ibest].factors[3])
# NTFk.plot2d(xfactor', ttu[ibest].factors[1]')
# NTFk.plot2d(yfactor', ttu[ibest].factors[2]')
# NTFk.plot2d(zfactor', ttu[ibest].factors[3]')
Mads.plotseries(ttu[ibest].factors[1], "figures-results/spnn-345-xfactors-estimated.png"; xaxis=0:19)
Mads.plotseries(ttu[ibest].factors[2], "figures-results/spnn-345-yfactors-estimated.png"; xaxis=0:49)
Mads.plotseries(ttu[ibest].factors[3], "figures-results/spnn-345-zfactors-estimated.png"; xaxis=0:49)

tcp, ecsize, ibest = NTFk.analysis(T_orig, [4]; prefix="results/tdcp-345")
NTFk.normalizelambdas!(tcp[ibest])
NTFk.normalizefactors!(tcp[ibest])
# NTFk.plot2matrices(xfactor, tcp[ibest].factors[1])
# NTFk.plot2matrices(yfactor, tcp[ibest].factors[2])
# NTFk.plot2matrices(zfactor, tcp[ibest].factors[3])
Mads.plotseries(xfactor)
Mads.plotseries(tcp[ibest].factors[1])
Mads.plotseries(yfactor)
Mads.plotseries(tcp[ibest].factors[2])
Mads.plotseries(zfactor)
Mads.plotseries(tcp[ibest].factors[3])

NTFk.plot3tensorsandcomponents(ttu[1], 1; maxvalue=2, xtitle="", timescale=false, ytitle="", movie=true, prefix="figures-results/spnn-345-d1", vspeed=10.0, order=[1,2,3])
NTFk.plot3tensorsandcomponents(ttu[1], 2; maxvalue=2, xtitle="", timescale=false, ytitle="", movie=true, prefix="figures-results/spnn-345-d2", vspeed=2., order=[1,2,3,4])
NTFk.plot3tensorsandcomponents(ttu[1], 3; maxvalue=2, xtitle="", timescale=false, ytitle="", movie=true, prefix="figures-results/spnn-345-d3", vspeed=2., order=[1,2,3,4,5])

NTFk.plot3tensorsandcomponents(ttu[1], 1; maxvalue=2, xtitle="", timescale=false, ytitle="",  prefix="figures-results/spnn-345-d1", maxcomponent=true, order=[1,2,3])
NTFk.plot3tensorsandcomponents(ttu[1], 2; maxvalue=2, xtitle="", timescale=false, ytitle="",  prefix="figures-results/spnn-345-d2", maxcomponent=true, order=[1,2,3,4])
NTFk.plot3tensorsandcomponents(ttu[1], 3; maxvalue=2, xtitle="", timescale=false, ytitle="",  prefix="figures-results/spnn-345-d3", maxcomponent=true, order=[1,2,3,4,5])