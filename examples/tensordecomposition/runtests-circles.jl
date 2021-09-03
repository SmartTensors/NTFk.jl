import NTFk
import TensorDecompositions
import Images
import ImageDraw

function makesignalcicrle(sx, sy, x, y, r)
	m = zeros(sx, sy)
	img = Images.Gray.(m)
	ImageDraw.draw!(img, ImageDraw.Ellipse(ImageDraw.CirclePointRadius(x, y, r; fill=true)))
	return convert(Array{Int64}, img)
end

tsize = (50, 50, 50)
T = Array{Float64}(undef, tsize)
v = [1]
xi = [0]
yi = [20]
ri = [8]
for t = 1:tsize[3]
	for i = 1:length(v)
		x = xi[i] + v[i] * t
		y = yi[i]
		r = ri[i]
		T[t,:,:] = makesignalcicrle(tsize[2], tsize[3], x, y, r)
	end
end

NTFk.plottensor(T; minvalue=0, maxvalue=1)
NTFk.plottensor(T; minvalue=0, maxvalue=1, movie=true, prefix="movies/circles-$(length(v))", quiet=true)

# CP analysis
tranks = [20]
tc, cp_s, cp_ibest = NTFk.analysis(T, tranks)
NTFk.plotcmptensors(T, tc[cp_ibest]; minvalue=0, maxvalue=1)
NTFk.plotcmptensors(T, tc[cp_ibest]; minvalue=0, maxvalue=1, movie=true, prefix="movies/circles-$(length(v))-cp", quiet=true)
NTFk.normalizefactors!(tc[cp_ibest])
NTFk.normalizelambdas!(tc[cp_ibest])
for i = 1:3
	NTFk.plotfactor(tc[cp_ibest], i)
end

# Tucker analysis
tt, tt_s, tt_ibest = NTFk.analysis(T, [tsize])
NTFk.plotcmptensors(T, tt[tt_ibest]; minvalue=0, maxvalue=1)
NTFk.plotcmptensors(T, tt[tt_ibest]; minvalue=0, maxvalue=1, movie=true, prefix="movies/circles-$(length(v))-tucker", quiet=true)
NTFk.normalizefactors!(tt[tt_ibest])
NTFk.normalizecore!(tt[tt_ibest])
for i = 1:3
	NTFk.plotfactor(tt[tt_ibest], i)
end

# Tucker analysis without a nonnegatity constaint on the core
tn, tn_s, tn_ibest = NTFk.analysis(T, [tsize]; core_nonneg=false)
NTFk.plotcmptensors(T, tn[tn_ibest]; minvalue=0, maxvalue=1)
NTFk.plotcmptensors(T, tn[tn_ibest]; minvalue=0, maxvalue=1, movie=true, prefix="movies/circles-$(length(v))-tucker-core", quiet=true)
NTFk.normalizefactors!(tn[tn_ibest])
NTFk.normalizecore!(tn[tn_ibest])
for i = 1:3
	NTFk.plotfactor(tn[tn_ibest], i)
end

# HOSVD analysis
th = NTFk.hosvd(T, tsize, [false,false,false])
NTFk.plotcmptensors(T, th; minvalue=0, maxvalue=1)
NTFk.plotcmptensors(T, th; minvalue=0, maxvalue=1, movie=true, prefix="movies/circles-$(length(v))-hosvd", quiet=true)
NTFk.normalizefactors!(th)
NTFk.normalizecore!(th)
for i = 1:3
	NTFk.plotfactor(th, i)
end