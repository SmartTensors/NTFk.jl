import NTFk

function makesignal(s, t, v)
	a = zeros(s, t)
	for i = 2:t-1
		k = convert(Int64, floor((i - 1) * v * t / s)) + 1
		if k > t - 1
			break
		else
			a[i, k] = 1
		end
	end
	return a
end

Random.seed!(10)

tsize = (50, 50, 50)
v = [1.1,1.2,1.3,1.4,1.5,1.6]
tt = Vector(length(v))
for i = 1:length(v)
	tt[i] = makesignal(tsize[1], tsize[3], v[i])
end
m = rand(vec(collect(0:length(v))), tsize[2])
T = Array{Float64}(undef, tsize)
for i = 1:tsize[2]
	if m[i] == 0
		T[:,i,:] = zeros(tsize[1], tsize[3])
	else
		T[:,i,:] = tt[m[i]]
	end
end
# NTFk.plottensor(T; movie=true, prefix="movies/signals-$(length(v))-50_50_50", quiet=true)
# NTFk.plottensor(T)

# tranks = [20]
# tc, c, ibest = NTFk.analysis(T, tranks; method=:cp_als)
# NTFk.plotcmptensors(T, tc[ibest]; minvalue=0, maxvalue=1000000)
# tt, c, ibest = NTFk.analysis(T, [tsize]; progressbar=true, max_iter=100000, lambda=1e-12)
# tt, c, ibest = NTFk.analysis(T, [tsize]; progressbar=true, core_nonneg=false)
# NTFk.plotcmptensors(T, tt[ibest]; minvalue=0, maxvalue=100)
th = TensorDecompositions.hosvd(T, tsize)
# NTFk.plotcmptensors(T, th; minvalue=0, maxvalue=1)
# NTFk.plotcmptensors(T, th; minvalue=0, maxvalue=1, movie=true, prefix="movies/signals-$(length(v))-50_50_50-cmp", quiet=true)
NTFk.normalizefactors!(th)
NTFk.normalizecore!(th)
ig = sortperm(NTFk.gettensorcomponentgroups(th, 2))
Te = TensorDecompositions.compose(th)
NTFk.plotcmptensors(T, Te[:, ig, :]; minvalue=0, maxvalue=1)
# NTFk.plotcmptensors(T, Te[:, ig, :]; minvalue=0, maxvalue=1, movie=true, prefix="movies/signals-$(length(v))-50_50_50-decomp", quiet=true)
