import PyPlot
import PyCall

function plotcube(A::Array; minvalue=NMFk.minimumnan(A), maxvalue=NMFk.maximumnan(A), nlevels=10, showaxes::Bool=false, showlegend::Bool=false, alpha::Number=1, cmap="RdYlGn", azim=120, elev=30, linewidth=0.8, filename="")
	nx, ny, nz = size(A)
	@show nx, ny, nz

	m3d = PyCall.pyimport("mpl_toolkits.mplot3d")

	PyPlot.matplotlib.rc("font", size=10)

	color = "gray"
	PyPlot.matplotlib.rc("text"; color=color)
	PyPlot.matplotlib.rc("axes"; labelcolor=color)
	PyPlot.matplotlib.rc("xtick"; color=color)
	PyPlot.matplotlib.rc("ytick"; color=color)

	PyPlot.clf()
	fig = PyPlot.figure(figsize=(8, 4))
	ax = m3d.Axes3D(fig; proj_type = "ortho")

	fig.patch.set_facecolor("none")
	fig.patch.set_alpha(0.0)
	ax = fig.gca(projection="3d")
	ax.view_init(elev, azim)
	ax.patch.set_facecolor("none")
	ax.patch.set_alpha(0.0)

	ax.set_xlim(1, nx)
	ax.set_ylim(1, ny)
	ax.set_zlim(1, nz)

	!showaxes && ax.axis("off")

	xgrid, ygrid = meshgrid(nx, ny)
	ax.contourf(xgrid, ygrid, A[:,:,nz], nlevels, vmin=minvalue, vmax=maxvalue, zdir="z", offset=nz, alpha=alpha, cmap=cmap)
	ax.plot_wireframe(xgrid, ygrid, ones(size(A[:,:,1]))*nz; color="gray", linewidth=linewidth)
	xgrid, zgrid = meshgrid(nx, nz)
	ax.contourf(xgrid, A[:,ny,:], zgrid, nlevels, vmin=minvalue, vmax=maxvalue, zdir="y", offset=ny, alpha=alpha, cmap=cmap)
	ax.plot_wireframe(xgrid, ones(size(A[:,ny,:]))*ny, zgrid; color="gray", linewidth=linewidth)
	ygrid, zgrid = meshgrid(ny, nz)
	ax.contourf(A[1,:,:], ygrid, zgrid, nlevels, vmin=minvalue, vmax=maxvalue, zdir="x", offset=1, alpha=alpha, cmap=cmap, zorder=0)
	ax.plot_wireframe(ones(size(A[1,:,:])), ygrid, zgrid; color="gray", linewidth=linewidth, zorder=10)

	showlegend && PyPlot.colorbar(cax)
	PyPlot.draw()
	filename != "" && PyPlot.savefig(filename)
	PyPlot.gcf()
end

function meshgrid(x::Vector, y::Vector)
	m = length(x)
	n = length(y)
	xx = reshape(x, 1, m)
	yy = reshape(y, n, 1)
	(repeat(xx, n, 1), repeat(yy, 1, m))
end
function meshgrid(nx::Number, ny::Number)
	x = collect(range(1, nx; length=nx))
	y = collect(range(1, ny; length=ny))
	xx = reshape(x, 1, nx)
	yy = reshape(y, ny, 1)
	(permutedims(repeat(xx, ny, 1)), permutedims(repeat(yy, 1, nx)))
end
@doc """
Create mesh grid

$(DocumentFunction.documentfunction(meshgrid;
argtext=Dict("x"=>"vector of grid x coordinates",
			"y"=>"vector of grid y coordinates")))

Returns:

- 2D grid coordinates based on the coordinates contained in vectors `x` and `y`
""" meshgrid