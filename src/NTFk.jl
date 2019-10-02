__precompile__()

"""
NTFk.jl: Non-negative Tensor Factorization + k-means clustering
---

NTFk performs unsupervised machine learning based on tensor decomposition coupled with sparsity and nonnegativity constraints.

NTFk methodology allows for automatic identification of the optimal number of features (signals) present in multi-dimensional data arrays (tensors).

The number of features (tensor \"rank\") along different dimensions can be estimated jointly and independently.

"""
module NTFk

import Pkg
using SharedArrays
using Printf
using SparseArrays
using Distributed

const ntfkdir = splitdir(splitdir(pathof(NTFk))[1])[1]

"Test NTFk functions"
function test()
	include(joinpath(ntfkdir, "test", "runtests.jl"))
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
import DocumentFunction
import Statistics
import LinearAlgebra
import Dates

modules = ["NTFk", "NMFk", "CanDecomp"]

include("NTFkHosvd.jl")
include("NTFkTucker.jl")
include("NTFkCP.jl")
include("NTFkNormalize.jl")
include("NTFkAtensor.jl")
include("NTFkDistributed.jl")
include("NTFkHelpers.jl")
include("NTFkHelp.jl")

include("NTFkLoadTensorDecompositions.jl")
include("NTFkTD-helpers.jl")
include("NTFkTD-memory.jl")

include("NTFkPlot.jl")
include("NTFkPlot2d.jl")
include("NTFkPlot3d.jl")
include("NTFkPlotMatrix.jl")
include("NTFkPlotComponents.jl")
include("NTFkPlotProgressBar.jl")
include("NTFkPlotColors.jl")

include("NTFkTensorly.jl")

include("NTFkGeo.jl")
include("NTFkAnalysis-seismic.jl")
include("NTFkAnalysis-wells.jl")
include("NTFkAnalysis-mixing.jl")

if haskey(ENV, "MATLAB_HOME")
	@tryimport MATLAB
else
	@info("MATLAB_HOME environmental variable is not defined!")
end

if isdefined(NTFk, :MATLAB)
	@info("MATLAB is available!")
	include("NTFkTensorToolBox.jl")
else
	@info("MATLAB is not installed!")
end

NTFk.welcome()

end