module dNTF

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

import CanDecomp
import TensorDecompositions
import TensorOperations
import TensorToolbox
import JLD
@tryimport MATLAB

const dntfdir = splitdir(Base.source_path())[1]

include("dntf-analysis.jl")
include("tensordecompositions-helpers.jl")
include("tensordecompositions-memory.jl")
include("tensordecompositions-analysis.jl")
include("tensor-display.jl")
if isdefined(:MATLAB)
    include("tensortoolbox-analysis.jl")
end

end