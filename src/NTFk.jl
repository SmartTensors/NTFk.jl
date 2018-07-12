# __precompile__()

"Non-negative Tensor Factorization + k-means Clustering"
module NTFk

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
include("NTFkPlot.jl")
include("NTFkAnalysis.jl")
include("NTFkAnalysis-tensordecompositions.jl")
include("NTFkAnalysis-tensordecompositions-helpers.jl")
include("NTFkAnalysis-tensordecompositions-memory.jl")
include("NTFkAnalysis-tensordecompositions-decomposition.jl")
if isdefined(:MATLAB)
    include("NTFkAnalysis-tensortoolbox-matlab.jl")
end

end