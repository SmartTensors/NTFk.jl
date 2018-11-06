__precompile__()

"Non-negative Tensor Factorization + k-means Clustering"
module NTFk

if VERSION >= v"0.7"
	import Pkg
	using SharedArrays
	using Printf
	using LinearAlgebra
	using SparseArrays
	using Distributed
	using Statistics
	using Dates
end

const ntfkdir = splitdir(splitdir(Base.source_path())[1])[1]

"Test NTFk functions"
function test()
	include(joinpath(ntfkdir, "test", "runtests.jl"))
end

"Checks if package is available"
function ispkgavailable(modulename::String; quiet::Bool=false)
	flag=false
	try
		Pkg.available(modulename)
		if typeof(Pkg.installed(modulename)) == Void
			flag=false
			!quiet && info("Module $modulename is not available")
		else
			flag=true
		end
	catch
		!quiet && info("Module $modulename is not available")
	end
	return flag
end

"Print error message"
function printerrormsg(errmsg::Any)
	Base.showerror(Base.STDERR, errmsg)
	if in(:msg, fieldnames(errmsg))
		warn(strip(errmsg.msg))
	elseif typeof(errmsg) <: AbstractString
		warn(errmsg)
	end
end

if VERSION >= v"0.7"
	"Try to import a module"
	macro tryimport(s::Symbol)
		mname = string(s)
		importq = string(:(import $s))
		infostring = string("Module ", s, " is not available")
		warnstring = string("Module ", s, " cannot be imported")
		q = quote
			try
				eval(Meta.parse($importq))
			catch errmsg
				printerrormsg(errmsg)
				warn($warnstring)
			end
		end
		return :($(esc(q)))
	end
else
	"Try to import a module"
	macro tryimport(s::Symbol)
		mname = string(s)
		importq = string(:(import $s))
		infostring = string("Module ", s, " is not available")
		warnstring = string("Module ", s, " cannot be imported")
		q = quote
			if ispkgavailable($mname; quiet=true)
				try
					eval(parse($importq))
				catch errmsg
					printerrormsg(errmsg)
					warn($warnstring)
				end
			else
				info($infostring)
			end
		end
		return :($(esc(q)))
	end
end

import NMFk
import CanDecomp
import TensorDecompositions
import TensorOperations
import TensorToolbox
import JLD
import PyPlot
import Gadfly
@tryimport MATLAB

modules = ["NTFk", "NMFk", "CanDecomp"]

const dntfdir = splitdir(Base.source_path())[1]

include("NTFkHelp.jl")
include("NTFkHelpers.jl")
include("NTFkPlot.jl")
include("NTFkAnalysis.jl")
include("NTFkAnalysis-TD.jl")
include("NTFkAnalysis-TD-helpers.jl")
include("NTFkAnalysis-TD-memory.jl")
include("NTFkAnalysis-TD-decomposition.jl")
include("NTFkAnalysis-tensorly.jl")

if VERSION >= v"0.7"
	ism = isdefined(NTFk, :MATLAB)
else
	ism = isdefined(:MATLAB)
end
if ism
	include("NTFkAnalysis-tensortoolbox.jl")
end

end