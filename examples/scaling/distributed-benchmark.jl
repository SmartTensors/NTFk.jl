import Distributed
Distributed.addprocs(4)

import LinearAlgebra
import BenchmarkTools
import DistributedArrays

@info("Set worker BLAS to one thread only")
@sync for p in Distributed.workers()
	@async Distributed.remotecall_wait(LinearAlgebra.BLAS.set_num_threads, p , 1)
end

const suite = BenchmarkTools.BenchmarkGroup()
suite["Local"] = BenchmarkTools.BenchmarkGroup()
suite["Distributed"] = BenchmarkTools.BenchmarkGroup()

function benchmark(T=Array, N=10)
	@BenchmarkTools.benchmarkable A * B setup = (A = $T(rand($N, $N)); B = $T(rand($N, $N)))
end

for N in (2^i for i = 5:13)
	suite["Local"][N] = benchmark(Array, N)
	suite["Distributed"][N] = benchmark(DistributedArrays.distribute, N)
end

function printresults(results)
	@info("Local")
	for (name, trial) in sort(collect(results["Local"]), by=x->time(x[2]))
		t = time(trial) / 1e6
		println(rpad(name, 25, "."), lpad(string(round(t, digits=2), " ms"), 20, "."))
	end
	@info("Distributed")
	for (name, trial) in sort(collect(results["Distributed"]), by=x->time(x[2]))
		t = time(trial) / 1e6
		println(rpad(name, 25, "."), lpad(string(round(t, digits=2), " ms"), 20, "."))
	end
end

BenchmarkTools.tune!(suite)
results = BenchmarkTools.run(suite)
printresults(results)