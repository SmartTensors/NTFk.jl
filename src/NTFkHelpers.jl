"Convert `@sprintf` macro into `sprintf` function"
sprintf(args...) = eval(:@sprintf($(args...)))

function maximumnan(X, c...; kw...)
    maximum(X[.!isnan.(X)], c...; kw...)
end

function minimumnan(X, c...; kw...)
    minimum(X[.!isnan.(X)], c...; kw...)
end

searchdir(key::Regex, path::String = ".") = filter(x->ismatch(key, x), readdir(path))
searchdir(key::String, path::String = ".") = filter(x->contains(x, key), readdir(path))

function getcsize(case::String; resultdir::String=".")
    files = searchdir(case, resultdir)
    csize = Array{Int64}(0, 3)
    kwa = Vector{String}(0)
    for (i, f) in enumerate(files)
        m = match(Regex(string("$(case)(.*)-([0-9]+)_([0-9]+)_([0-9]+).jld")), f)
        if m != nothing
            push!(kwa, m.captures[1])
            c = parse.(Int64, m.captures[2:end])
            csize = vcat(csize, c')
        end
    end
    return csize, kwa
end

function gettensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1; core::Bool=false)
    cs = size(t.core)[dim]
    csize = TensorToolbox.mrank(t.core)
    ndimensons = length(csize)
    @assert dim >= 1 && dim <= ndimensons
    crank = csize[dim]
    Xe = Vector{Any}(cs)
    tt = deepcopy(t)
    for i = 1:cs
        if core
            for j = 1:cs
                if i !== j
                    nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
                    tt.core[nt...] .= 0
                end
            end
            Xe[i] = TensorDecompositions.compose(tt)
            tt.core .= t.core
        else
            for j = 1:cs
                if i !== j
                    tt.factors[dim][:, j] .= 0
                end
            end
            Xe[i] = TensorDecompositions.compose(tt)
            tt.factors[dim] .= t.factors[dim]
        end
    end
    m = maximum.(Xe)
    imax = sortperm(m; rev=true)
    return Xe[imax[1:crank]]
end

function getgridvalues(v, r; logtransform=true)
    lv = length(v)
    lr = length(r)
    @assert lv == lr
    f = similar(v)
    for i=1:lv
        try
            if logtransform
                f[i] = Interpolations.interpolate((log10.(r[i]),), 1:length(r[i]), Interpolations.Gridded(Interpolations.Linear()))[log10(v[i])]
            else
                f[i] = Interpolations.interpolate((r[i],), 1:length(r[i]), Interpolations.Gridded(Interpolations.Linear()))[v[i]]
            end
        catch
            if logtransform
                f[i] = Interpolations.interpolate((sort!(log10.(r[i])),), length(r[i]):-1:1, Interpolations.Gridded(Interpolations.Linear()))[log10(v[i])]
            else
                f[i] = Interpolations.interpolate((sort!(r[i]),), length(r[i]):-1:1, Interpolations.Gridded(Interpolations.Linear()))[v[i]]
            end
        end
    end
    return f
end

function getinterpolatedtensor(t::TensorDecompositions.Tucker{T,N}, v; sp=[Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line())), Interpolations.OnCell()]) where {T,N}
    lv = length(v)
    f = Vector(lv)
    factors = []
    for i = 1:N
        push!(factors, t.factors[i])
    end
    for j = 1:lv
        if !isnan(v[j])
            cv = size(t.factors[j], 2)
            f = Array{T}(1,cv)
            for i = 1:cv
                f[1, i] = Interpolations.interpolate(t.factors[j][:, i], sp...)[v[j]]
            end
            factors[j] = f
        end
    end
    tn = TensorDecompositions.Tucker((factors...), t.core)
    return tn
end

function gettensorcomponentorder(t::TensorDecompositions.Tucker, dim::Integer=1; method::Symbol=:core, firstpeak::Bool=true, quiet=true)
    cs = size(t.core)[dim]
    csize = TensorToolbox.mrank(t.core)
    ndimensons = length(csize)
    @assert dim >= 1 && dim <= ndimensons
    crank = csize[dim]
    if method == :factormagnitude
        fmin = vec(minimum(t.factors[dim], 1))
        fmax = vec(maximum(t.factors[dim], 1))
        @assert cs == length(fmax)
        fdx = fmax .- fmin
        for i = 1:cs
            if fmax[i] == 0
                warn("Maximum of component $i is equal to zero!")
            end
            if fdx[i] == 0
                warn("Component $i has zero variability!")
            end
        end
        ifdx = sortperm(fdx; rev=true)[1:crank]
        !quiet && info("Factor magnitudes (max - min): $fdx")
        if firstpeak
            imax = map(i->indmax(t.factors[dim][:, ifdx[i]]), 1:crank)
            order = ifdx[sortperm(imax)]
        else
            order = ifdx[1:crank]
        end
    else
        maxXe = Vector{Float64}(cs)
        tt = deepcopy(t)
        for i = 1:cs
            if method == :core
                for j = 1:cs
                    if i !== j
                        nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
                        tt.core[nt...] .= 0
                    end
                end
                Te = TensorDecompositions.compose(tt)
                maxXe[i] = maximum(Te) - minimum(Te)
                tt.core .= t.core
            else
                for j = 1:cs
                    if i !== j
                        tt.factors[dim][:, j] .= 0
                    end
                end
                Te = TensorDecompositions.compose(tt)
                maxXe[i] = maximum(Te) - minimum(Te)
                tt.factors[dim] .= t.factors[dim]
            end
        end
        !quiet && info("Max core magnitudes: $maxXe")
        imax = sortperm(maxXe; rev=true)
        order = imax[1:crank]
    end
    return order
end

function gettensorminmax(t::TensorDecompositions.Tucker, dim::Integer=1; method::Symbol=:core)
    cs = size(t.core)[dim]
    csize = TensorToolbox.mrank(t.core)
    ndimensons = length(csize)
    @assert dim >= 1 && dim <= ndimensons
    crank = csize[dim]
    if method == :factormagnitude
        fmin = vec(minimum(t.factors[dim], 1))
        fmax = vec(maximum(t.factors[dim], 1))
        @assert cs == length(fmax)
        for i = 1:cs
            if fmax[i] == 0
                warn("Maximum of component $i is equal to zero!")
            end
        end
        info("Max factor magnitudes: $fmax")
        info("Min factor magnitudes: $fmin")
    elseif method == :all
        Te = TensorDecompositions.compose(t)
        tsize = size(Te)
        ts = tsize[dim]
        maxTe = Vector{Float64}(ts)
        minTe = Vector{Float64}(ts)
        for i = 1:tsize[dim]
            nt = ntuple(k->(k == dim ? i : Colon()), ndimensons)
            minTe[i] = minimum(Te[nt...])
            maxTe[i] = maximum(Te[nt...])
        end
        info("Max all magnitudes: $maxTe")
        info("Min all magnitudes: $minTe")
    else
        maxXe = Vector{Float64}(cs)
        minXe = Vector{Float64}(cs)
        tt = deepcopy(t)
        for i = 1:cs
            if method == :core
                for j = 1:cs
                    if i !== j
                        nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
                        tt.core[nt...] .= 0
                    end
                end
                Te = TensorDecompositions.compose(tt)
                minXe[i] = minimum(Te)
                maxXe[i] = maximum(Te)
                tt.core .= t.core
            else
                for j = 1:cs
                    if i !== j
                        tt.factors[dim][:, j] .= 0
                    end
                end
                Te = TensorDecompositions.compose(tt)
                minXe[i] = minimum(Te)
                maxXe[i] = maximum(Te)
                tt.factors[dim] .= t.factors[dim]
            end
        end
        info("Max core magnitudes: $maxXe")
        info("Min core magnitudes: $minXe")
    end
end

function gettensorcomponentgroups(t::TensorDecompositions.Tucker, dim::Integer=1; cutvalue::Number=0.9)
    g = zeros(t.factors[dim][:, 1])
    v = maximum(t.factors[dim], 1) .> cutvalue
    gi = 0
    for i = 1:length(v)
        if v[i]
            m = t.factors[dim][:, i] .> cutvalue
            gi += 1
            g[m] = gi
        end
    end
    info("Number of component groups in dimension $dim is $(gi)")
    return g
end
