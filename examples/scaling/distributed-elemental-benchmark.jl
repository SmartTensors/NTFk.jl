import Distributed
using MPI
# --prefix /home/vvv/.linuxbrew-mads/bin
man = MPIManager(np=2, mpirun_cmd=`mpiexec  -np 2 -host es10,es12`)
Distributed.addprocs()
Distributed.addprocs(4)
Distributed.addprocs(["madsmin", "mads12", "mads14"])

Distributed.addprocs([fill("cn4020", 48); fill("cn4021", 48)])

import LinearAlgebra
import BenchmarkTools
import DistributedArrays
import Elemental
import CSV

function printresults(results)
	@info("Elemental Local: Size Time [ms]")
	for (name, trial) in sort(collect(results["Elemental Local"]), by=x->x[1])
		t = time(trial) / 1e6
		println(rpad(name, 25, "_"), lpad(string(round(t, digits=2), ""), 20, "_"))
	end
	@info("Elemental Distributed: Size Time [ms]")
	for (name, trial) in sort(collect(results["Elemental Distributed"]), by=x->x[1])
		t = time(trial) / 1e6
		println(rpad(name, 25, "_"), lpad(string(round(t, digits=2), ""), 20, "_"))
	end
end

function setbenchmark(A=Array, T=Float32, N=10; nsamples=10, nevals=1, nseconds=60)
	@BenchmarkTools.benchmarkable A * B setup=(A = $A($T); Elemental.gaussian!(A, $N, $N); B = $A($T); Elemental.gaussian!(B, $N, $N)) samples=nsamples evals=nevals seconds=nseconds
end

function setbenchmark(r::Integer=7, T=Float32; nsamples=10, nevals=1, nseconds=60)
	for N in (2^i for i = 5:r)
		suite["Elemental Local"][N] = setbenchmark(Elemental.Matrix, T, N; nsamples=nsamples, nevals=nevals, nseconds=nseconds)
		suite["Elemental Distributed"][N] = setbenchmark(Elemental.DistMatrix, T, N; nsamples=nsamples, nevals=nevals, nseconds=nseconds)
	end
end

@info("Set worker BLAS to one thread only")
@sync for p in Distributed.workers()
	@async Distributed.remotecall_wait(LinearAlgebra.BLAS.set_num_threads, p , 1)
end

@info("Turn off BLAS thread")
@sync for p in Distributed.workers()
	@async Distributed.remotecall_wait(LinearAlgebra.BLAS.set_num_threads, p, -1)
end

suite = BenchmarkTools.BenchmarkGroup()
suite["Elemental Local"] = BenchmarkTools.BenchmarkGroup()
suite["Elemental Distributed"] = BenchmarkTools.BenchmarkGroup()

setbenchmark(13, Float32; nsamples=100, nevals=1, nseconds=60)
BenchmarkTools.tune!(suite)
results = BenchmarkTools.run(suite)
printresults(results)

colors = ["red", "blue", "green", "orange", "magenta", "cyan", "brown", "pink", "lime", "navy", "maroon", "yellow", "olive", "springgreen", "teal", "coral", "#e6beff", "beige", "purple", "#4B6F44", "#9F4576"]
l = []
d = CSV.read("local-cross.txt"; ignorerepeated=true, comment="[", delim='_', header=[:Size, :Time])
push!(l, Gadfly.layer(x=d[:Size], y=d[:Time], Gadfly.Geom.line, Gadfly.Geom.point, Gadfly.Theme(default_color="red", line_width=2.5Gadfly.pt, point_size=3.5Gadfly.pt, highlight_width=0Gadfly.pt)))
for (i, f) in enumerate(["2nodes-cross.txt", "3nodes-cross.txt", "4nodes-cross.txt"])
	d = CSV.read(f; ignorerepeated=true, comment="[", delim='_', header=[:Size, :Time])
	push!(l, Gadfly.layer(x=d[:Size], y=d[:Time], Gadfly.Geom.line, Gadfly.Geom.point, Gadfly.Theme(default_color=colors[i+1], line_width=2.5Gadfly.pt, point_size=3.5Gadfly.pt, highlight_width=0Gadfly.pt)))
end
# Gadfly.Guide.title("Performance: Cross"),
p = Gadfly.plot(l..., Gadfly.Guide.XLabel("Matrix Size [-]"), Gadfly.Guide.YLabel("Time [ms]"), Gadfly.Coord.cartesian(xmin=1, xmax=4), Gadfly.Scale.x_log10(), Gadfly.Scale.y_log10(), Gadfly.Guide.manual_color_key("", ["Local", "2 nodes", "3 nodes", "4 nodes"], colors[1:4]), Gadfly.Theme(major_label_font_size=16Gadfly.pt, key_label_font_size=14Gadfly.pt, minor_label_font_size=8Gadfly.pt))
Gadfly.draw(Gadfly.PNG("scaling.png", 12Gadfly.inch, 6Gadfly.inch, dpi=300), p)