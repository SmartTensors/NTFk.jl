using Distributed
Distributed.addprocs(4)

import DistributedArrays

@everywhere begin
	import Random
	import DistributedArrays
	Random.seed!(Distributed.myid())
	r = rand(10)
end

rAs = [@spawnat p rand(N) for p in workers()]

DistributedArrays.DArray(rAs)

a = rand(1:5, 1_000);
b = rand(1_000);

@time da = DistributedArrays.distribute(a);
@time db = DistributedArrays.distribute(b);

@time sum(a .* b)
@time sum(da .* db)

import Pkg; Pkg.add("BenchmarkTools")
@everywhere using DistributedArrays
@everywhere using ParallelDataTransfer
@eval @everywhere s = sum(DistributedArrays.localpart($da) .* DistributedArrays.localpart($db))
global s = 0
for w in workers()
	global s += getfrom(w, :s)
end
@show s

function life_step(d::DArray)
    DArray(size(d),procs(d)) do I
    	@show I
    end
end