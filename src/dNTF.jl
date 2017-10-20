module dNTF

import TensorDecompositions
import TensorOperations
import Combinatorics

const dntfdir = splitdir(Base.source_path())[1]

include("tensordecompositions-helpers.jl")
include("tensordecompositions-memory.jl")
include("tensor-display.jl")
include("tensor-analysis.jl")

end