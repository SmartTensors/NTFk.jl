function seismicdensity(t, dim, cx, cy, x0, y0; nbin=20, xmatrix=nothing, ymatrix=nothing)
	Xs = NTFk.gettensorslices(t, dim; mask=1)
	sy, sx = size(Xs[1])
	fieldx = x0
	fieldy = y0
	pr = ["T$i" for i=1:length(Xs)]
	ll = []
	for s = 1:length(Xs)
		display(NTFk.plotmatrix(Xs[s]; dots=[fieldx fieldy;], dotsize=10Gadfly.pt, xmatrix=xmatrix, ymatrix=ymatrix))
		fielddst = Vector{Float64}(undef, 0)
		fieldmag = Vector{Float64}(undef, 0)
		# fieldind = []
		for i = 1:sy
			dx = (cy[i] - fieldy)^2
			for j = 1:sx
				if !isnan(Xs[s][i, j])
					push!(fielddst, sqrt(dx + (cx[j] - fieldx)^2))
					push!(fieldmag, Xs[s][i, j])
					# push!(fieldind, (i,j))
				end
			end
		end
		# display(Gadfly.plot(x=fielddst, y=fieldmag, Gadfly.Scale.x_log10(), Gadfly.Scale.y_log10()))
		is = sortperm(fielddst)
		logdstrange = range(log10(minimum(fielddst)), log10(maximum(fielddst)); length=nbin+1)
		dstrange = 10 .^ logdstrange
		dstrange[end] = maximum(fielddst)
		fielddstbin = Vector{Float64}(undef, nbin)
		fieldmagbin = Vector{Float64}(undef, nbin)
		fieldcntbin = zeros(nbin)
		fieldcntbinn = Vector{Float64}(undef, nbin)
		id = 1
		rold = dstrange[1]
		nevents = length(fieldmag)
		for (b, r) in enumerate(dstrange[2:end])
			fieldcntbin[b] = 0
			fieldmagbin[b] = 0
			while fielddst[is][id] <= r
				fieldcntbin[b] += 1
				fieldmagbin[b] += fieldmag[is][id]
				# Xs[s][fieldind[is][id]...] = 10
				id += 1
				if id > nevents
					break
				end
			end
			# display(NTFk.plotmatrix(Xs[s]; dots=[fieldx fieldy;], dotsize=10Gadfly.pt, xmatrix=[minx, maxx], ymatrix=[miny, maxy]))
			# @show fieldcntbin[b]
			radius = (r + rold) / 2
			if fieldcntbin[b] == 0
				fieldcntbinn[b] = NaN
				fieldmagbin[b] = NaN
			else
				A = 2 * pi * radius * (r - rold)
				fieldcntbinn[b] = fieldcntbin[b] / A
				fieldmagbin[b] /= A
			end
			fielddstbin[b] = (r + rold) /2
			rold = r
		end
		@assert length(fieldmag) == sum(fieldcntbin)
		l = Gadfly.layer(x=fielddstbin, y=fieldcntbinn, Gadfly.Geom.line(), Gadfly.Geom.point(), Gadfly.Theme(line_width=3Gadfly.pt, point_size=2Gadfly.pt, highlight_width=0Gadfly.pt, default_color=parse(Gadfly.Colorant, colors[s])))
		push!(ll, l)
		# display(Gadfly.plot(l, Gadfly.Guide.XLabel("log10 Density"), Gadfly.Guide.YLabel("log10 Distance"), Gadfly.Scale.x_log10(), Gadfly.Scale.y_log10()))
		# display(Gadfly.plot(x=fielddstbin, y=fieldmagbin, Gadfly.Guide.XLabel("log10 Density"), Gadfly.Guide.YLabel("log10 Distance"), Gadfly.Geom.line(), Gadfly.Geom.point(), Gadfly.Theme(line_width=4Gadfly.pt, point_size=5Gadfly.pt, default_color=parse(Gadfly.Colorant, colors[s])), Gadfly.Scale.x_log10(), Gadfly.Scale.y_log10()))
	end
	display(Gadfly.plot(ll..., Gadfly.Guide.XLabel("log10 Density"), Gadfly.Guide.YLabel("log10 Distance"), Gadfly.Scale.x_log10(), Gadfly.Scale.y_log10(), Gadfly.Guide.manual_color_key("", pr, colors[1:length(pr)])))
end