__precompile__()

"Non-negative Tensor Factorization + k-means Clustering and sparsity constraints"
module NTFk

import Pkg
import Dates
using SharedArrays
using Printf
using LinearAlgebra
using SparseArrays
using Distributed
using Statistics

"Test NTFk functions"
function test()
	include(joinpath(ntfkdir, "test", "runtests.jl"))
end

"Checks if package is available"
function ispkgavailable(modulename::String; quiet::Bool=false)
	flag=false
	try
		Pkg.available(modulename)
		if typeof(Pkg.installed(modulename)) == Nothing
			flag=false
			!quiet && @info("Module $modulename is not available")
		else
			flag=true
		end
	catch
		!quiet && @info("Module $modulename is not available")
	end
	return flag
end

"Print error message"
function printerrormsg(errmsg::Any)
	Base.showerror(Base.stderr, errmsg)
	try
		if in(:msg, fieldnames(errmsg))
			@warn(strip(errmsg.msg))
		elseif typeof(errmsg) <: AbstractString
			@warn(errmsg)
		end
	catch
		@warn(errmsg)
	end
end

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
			@warn($warnstring)
		end
	end
	return :($(esc(q)))
end

global imagedpi = 300
global DMAXITER = 1000
global outputformat = "jld2"

import NMFk
import CanDecomp
import TensorDecompositions
import TensorOperations
import TensorToolbox
import JLD
import JLD2
import FileIO
import PyPlot
import Gadfly

modules = ["NTFk", "NMFk", "CanDecomp"]

const ntfkdir = splitdir(splitdir(pathof(NTFk))[1])[1]

include("NTFkHelp.jl")
include("NTFkHelpers.jl")
include("NTFkPlotColors.jl")
include("NTFkPlot2d.jl")
include("NTFkPlotMatrix.jl")
include("NTFkPlotProgressBar.jl")
include("NTFkPlot.jl")
include("NTFkPlot3D.jl")
include("NTFkPlotComponents.jl")
include("NTFkGeo.jl")
include("NTFkSeismic.jl")
include("NTFkAnalysis.jl")
include("NTFkAnalysis-mixing.jl")
include("NTFkAnalysis-helpers.jl")
include("NTFkAnalysis-memory.jl")
include("NTFkAnalysis-normalize.jl")
include("NTFkAnalysis-atensor.jl")
include("NTFkAnalysis-tucker.jl")
include("NTFkAnalysis-cp.jl")
include("NTFkAnalysis-hosvd.jl")
include("NTFkAnalysis-tensorly.jl")
include("NTFkLoadTensorDecompositions.jl")

if haskey(ENV, "MATLAB_HOME")
	@tryimport MATLAB
else
	@info("MATLAB_HOME environmental variable is not defined!")
end

if isdefined(NTFk, :MATLAB)
	@info("MATLAB is available!")
	include("NTFkAnalysis-tensortoolbox.jl")
else
	@info("MATLAB is not installed!")
end

end