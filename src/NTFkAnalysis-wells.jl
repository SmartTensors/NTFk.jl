import NMFk
import Statistics

function computestats(X, Xe, volumeindex=1:size(Xe,1), wellindex=1:size(Xe,3), timeindex=:, c=""; plot::Bool=false, quiet::Bool=true, wellnames=nothing, xaxis=1:size(Xe,2))
	m = "%-85s"
	if Xe == nothing
		@info("$(NMFk.sprintf(m, c)): fails")
		return
	end
	ferr = Array{Float64}(undef, length(volumeindex))
	wsum1 = Array{Float64}(undef, length(wellindex))
	wsum2 = Array{Float64}(undef, length(wellindex))
	werr = Array{Float64}(undef, length(wellindex))
	merr = Array{Float64}(undef, length(volumeindex))
	fcor = Array{Float64}(undef, length(volumeindex))
	g = "%.2g"
	f = "%.2f"
	for (i, v) in enumerate(volumeindex)
		# se = NMFk.sumnan(Xe[wellindex,timeindex,v])
		# s =  NMFk.sumnan(X[wellindex,timeindex,v])
		# @show se
		# @show s
		# ferr[i] = (se - s) / s
		ferr[i] = NMFk.normnan(X[wellindex,timeindex,v] .- Xe[wellindex,timeindex,v])
		for (j, w) in enumerate(wellindex)
			wsum2[j] = NMFk.sumnan(Xe[w,timeindex,v])
			wsum1[j] = NMFk.sumnan(X[w,timeindex,v])
			werr[j] = abs.(wsum2[j] - wsum1[j]) / wsum1[j]
		end
		merr[i] = maximum(werr)
		fcor[i] = Statistics.cor(vec(wsum1), vec(wsum2))
	end
	namecase = lowercase(replace(replace(c, " "=>"_"), "/"=> "_"))
	# @info("$(NMFk.sprintf(m, c)): $(NMFk.sprintf(f, NMFk.normnan(X[wellindex,timeindex,volumeindex] .- Xe[wellindex,timeindex,volumeindex]))) [Error: $(NMFk.sprintf(g, ferr[1])) $(NMFk.sprintf(g, ferr[2])) $(NMFk.sprintf(g, ferr[3]))] [Max error: $(NMFk.sprintf(g, merr[1])) $(NMFk.sprintf(g, merr[2])) $(NMFk.sprintf(g, merr[3]))] [Pearson: $(NMFk.sprintf(g, fcor[1])) $(NMFk.sprintf(g, fcor[2])) $(NMFk.sprintf(g, fcor[3]))]")
	@info("$(NMFk.sprintf(m, c)): $(NMFk.sprintf(f, NMFk.normnan(X[wellindex,timeindex,volumeindex] .- Xe[wellindex,timeindex,volumeindex]))) : $(NMFk.sprintf(f, ferr[1])) : $(NMFk.sprintf(f, ferr[2])) : $(NMFk.sprintf(f, ferr[3])) : $(NMFk.sprintf(g, fcor[1])) : $(NMFk.sprintf(g, fcor[2])) : $(NMFk.sprintf(g, fcor[3]))")
	if !plot
		return nothing
	else
		return NTFk.plot2d(X, Xe; quiet=quiet, figuredir="results-12-18", keyword=namecase, titletext=c, wellnames=wellnames, dimname="Well", xaxis=xaxis, ymin=0, xmin=Dates.Date(2015,12,15), xmax=Dates.Date(2017,6,10), linewidth=1.5Gadfly.pt, gm=[Gadfly.Guide.manual_color_key("", ["Oil", "Gas", "Water"], ["green", "red", "blue"])], colors=["green", "red", "blue"])
	end
end