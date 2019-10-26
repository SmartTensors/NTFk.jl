import DocumentFunction

function welcome()
	c = Base.text_colors
	tx = c[:normal] # text
	bl = c[:bold] # bold
	d1 = c[:bold] * c[:blue]    # first dot
	d2 = c[:bold] * c[:red]     # second dot
	d3 = c[:bold] * c[:green]   # third dot
	d4 = c[:bold] * c[:magenta] # fourth dot
	println("$(bl)NTFk: Nonnegative Tensor Factorization + k-means clustering$(tx)")
	println("====")
	println("")
	println("$(d1)  _     _  $(d2) _________ $(d3)  _______   $(d4)_$(tx)")
	println("$(d1) |  \\  | | $(d2)|___   ___| $(d3)|  _____| $(d4)| |  _$(tx)")
	println("$(d1) | . \\ | | $(d2)    | |     $(d3)| |___    $(d4)| | / /$(tx)")
	println("$(d1) | |\\ \\| | $(d2)    | |     $(d3)|  ___|   $(d4)| |/ /$(tx)")
	println("$(d1) | | \\ ' | $(d2)    | |     $(d3)| |       $(d4)|   ($(tx)")
	println("$(d1) | |  \\  | $(d2)    | |     $(d3)| |       $(d4)| |\\ \\$(tx)")
	println("$(d1) |_|   \\_| $(d2)    |_|     $(d3)|_|       $(d4)|_| \\_\\$(tx)")
	println("")
	println("NTFk performs unsupervised machine learning based on tensor decomposition coupled with sparsity and nonnegativity constraints.")
	println("NTFk methodology allows for automatic identification of the optimal number of features (signals) present in multi-dimensional data arrays (tensors).")
	println("The number of features (tensor \"rank\") along different dimensions can be estimated jointly and independently.")
end

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
List available functions in the NTFk modules:

$(DocumentFunction.documentfunction(functions;
argtext=Dict("string"=>"string to display functions with matching names",
			"m"=>"NTFk module")))

Examples:

```julia
NTFk.functions()
NTFk.functions("get")
NTFk.functions(NTFk, "get")
```
""" functions

"Checks if package is available"
function ispkgavailable(modulename::String)
	haskey(Pkg.installed(), modulename)
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
macro tryimport(s::Symbol, domains::Symbol=:NTFk)
	mname = string(s)
	domain = eval(domains)
	if !ispkgavailable(mname)
		try
			Pkg.add(mname)
		catch
			@info string("Module ", s, " is not available!")
			return nothing
		end
	end
	if !isdefined(domain, s)
		importq = string(:(import $s))
		warnstring = string("Module ", s, " cannot be imported!")
		q = quote
			try
				Core.eval($domain, Meta.parse($importq))
			catch errmsg
				printerrormsg(errmsg)
				@warn($warnstring)
			end
		end
		return :($(esc(q)))
	end
end
