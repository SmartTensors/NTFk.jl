import DocumentFunction
import NTFk

function functions(re::Regex; stdout::Bool=false, quiet::Bool=false)
	n = 0
	for i in modules
		Core.eval(NTFk, :(@tryimport $(Symbol(i))))
		n += functions(Symbol(i), re; stdout=stdout, quiet=quiet)
	end
	n > 0 && string == "" && @info("Total number of functions: $n")
	return
end
function functions(string::String=""; stdout::Bool=false, quiet::Bool=false)
	n = 0
	for i in modules
		Core.eval(NTFk, :(@tryimport $(Symbol(i))))
		n += functions(Symbol(i), string; stdout=stdout, quiet=quiet)
	end
	n > 0 && string == "" && @info("Total number of functions: $n")
	return
end
function functions(m::Union{Symbol, Module}, re::Regex; stdout::Bool=false, quiet::Bool=false)
	n = 0
	try
		f = names(eval(m), true)
		functions = Array{String}(undef, 0)
		for i in 1:length(f)
			functionname = "$(f[i])"
			if occursin("eval", functionname) || occursin("#", functionname) || occursin("__", functionname) || functionname == "$m"
				continue
			end
			if ismatch(re, functionname)
				push!(functions, functionname)
			end
		end
		if length(functions) > 0
			!quiet && @info("$(m) functions:")
			sort!(functions)
			n = length(functions)
			if stdout
				!quiet && Base.display(TextDisplay(STDOUT), functions)
			else
				!quiet && Base.display(functions)
			end
		end
	catch
		@warn("Module $m not defined!")
	end
	n > 0 && string == "" && @info("Number of functions in module $m: $n")
	return n
end
function functions(m::Union{Symbol, Module}, string::String=""; stdout::Bool=false, quiet::Bool=false)
	n = 0
	if string != ""
		quiet=false
	end
	try
		f = names(Core.eval(NTFk, m); all=true)
		functions = Array{String}(undef, 0)
		for i in 1:length(f)
			functionname = "$(f[i])"
			if occursin("eval", functionname) || occursin("#", functionname) || occursin("__", functionname) || functionname == "$m"
				continue
			end
			if string == "" || occursin(string, functionname)
				push!(functions, functionname)
			end
		end
		if length(functions) > 0
			!quiet && @info("$(m) functions:")
			sort!(functions)
			n = length(functions)
			if stdout
				!quiet && Base.display(TextDisplay(STDOUT), functions)
			else
				!quiet && Base.display(functions)
			end
		end
	catch
		@warn("Module $m not defined!")
	end
	n > 0 && string == "" && @info("Number of functions in module $m: $n")
	return n
end

@doc """
List available functions in the MADS modules:

$(DocumentFunction.documentfunction(functions;
argtext=Dict("string"=>"string to display functions with matching names",
			"m"=>"MADS module")))

Examples:

```julia
NTFk.functions()
NTFk.functions(BIGUQ)
NTFk.functions("get")
NTFk.functions(NTFk, "get")
```
""" functions