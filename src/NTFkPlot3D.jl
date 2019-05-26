import PyPlot
import PyCall

function plotcube(A::Array; minvalue=minimumnan(A), maxvalue=maximumnan(A), nlevels=10, showaxes::Bool=false, showlegend::Bool=false, alpha::Number=1, cmap="RdYlGn", azim=-60, elev=30)
	nx, ny, nz = size(A)

	m3d = PyCall.pyimport("mpl_toolkits.mplot3d")

	PyPlot.matplotlib.rc("font", size=10)

	COLOR = "gray"
	PyPlot.matplotlib.rc("text"; color=COLOR)
	PyPlot.matplotlib.rc("axes"; labelcolor=COLOR)
	PyPlot.matplotlib.rc("xtick"; color=COLOR)
	PyPlot.matplotlib.rc("ytick"; color=COLOR)

	PyPlot.clf()
	fig = PyPlot.figure()
	ax = m3d.Axes3D(fig)

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
	xgrid, zgrid = meshgrid(nx, nz)
	ax.contourf(xgrid, A[:,1,:], zgrid, nlevels, vmin=minvalue, vmax=maxvalue, zdir="y", offset=1, alpha=alpha, cmap=cmap)
	ygrid, zgrid = meshgrid(ny, nz)
	cax = ax.contourf(A[nx,:,:], ygrid, zgrid, nlevels, vmin=minvalue, vmax=maxvalue, zdir="x", offset=nx, alpha=alpha, cmap=cmap)

	showlegend && PyPlot.colorbar(cax)
	PyPlot.draw()
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