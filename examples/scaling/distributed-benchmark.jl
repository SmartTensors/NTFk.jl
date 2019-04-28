import Distributed
Distributed.addprocs(4)

import LinearAlgebra
import BenchmarkTools
import DistributedArrays

function printresults(results)
	@info("Local")
	for (name, trial) in sort(collect(results["Local"]), by=x->x[1])
		t = time(trial) / 1e6
		println(rpad(name, 25, "."), lpad(string(round(t, digits=2), " ms"), 20, "."))
	end
	@info("Distributed")
	for (name, trial) in sort(collect(results["Distributed"]), by=x->x[1])
		t = time(trial) / 1e6
		println(rpad(name, 25, "."), lpad(string(round(t, digits=2), " ms"), 20, "."))
	end
end

function setbenchmark(A=Array, T=Float32, N=10; nsamples=10, nevals=1, nseconds=60)
	@BenchmarkTools.benchmarkable A * B setup=(A = $A(rand($T, $N, $N)); B = $A(rand($T, $N, $N))) samples=nsamples evals=nevals seconds=nseconds
end

function setbenchmark(r::Integer=7, T=Float32; nsamples=10, nevals=1, nseconds=60)
	for N in (2^i for i = 5:r)
		suite["Local"][N] = setbenchmark(Array, T, N; nsamples=nsamples, nevals=nevals, nseconds=nseconds)
		suite["Distributed"][N] = setbenchmark(DistributedArrays.distribute, T, N; nsamples=nsamples, nevals=nevals, nseconds=nseconds)
	end
end

# @info("Set worker BLAS to one thread only")
# @sync for p in Distributed.workers()
# 	@async Distributed.remotecall_wait(LinearAlgebra.BLAS.set_num_threads, p , 1)
# end

suite = BenchmarkTools.BenchmarkGroup()
suite["Local"] = BenchmarkTools.BenchmarkGroup()
suite["Distributed"] = BenchmarkTools.BenchmarkGroup()

setbenchmark(13, Float32; nsamples=100, nevals=1, nseconds=60)
BenchmarkTools.tune!(suite)
results = BenchmarkTools.run(suite)
printresults(results)