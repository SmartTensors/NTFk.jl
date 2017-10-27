module dNTF

import TensorDecompositions
import TensorOperations
import TensorToolbox
import JLD

const dntfdir = splitdir(Base.source_path())[1]

include("tensordecompositions-helpers.jl")
include("tensordecompositions-memory.jl")
include("tensordecompositions-analysis.jl")
include("tensortoolbox-analysis.jl")
include("tensor-display.jl")

end