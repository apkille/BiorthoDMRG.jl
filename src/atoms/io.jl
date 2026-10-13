const BACKENDS = Dict{String,Type}("Biortho" => Biortho)

_scalar(x::Number) = x
_scalar(x::Bool) = Int(x)

_write(g, k::String, ::Nothing) = nothing
_write(g, k::String, v::Number) = (g[k] = _scalar(v))
_write(g, k::String, v::AbstractArray{<:Number}) = (g[k] = collect(v))
_write(g, k::String, v::AbstractString) = (g[k] = String(v))
_write(g, k::String, v::Union{Solution,NamedTuple}) = _write(HDF5.create_group(g, k), v)

function _write(g, s::Solution)
    a = HDF5.attrs(g)
    a["kind"] = "Solution"
    a["backend"] = string(nameof(typeof(s.backend)))
    for f in fieldnames(typeof(s.backend))
        a[String(f)] = _scalar(getfield(s.backend, f))
    end
    g["t"] = s.t
    g["names"] = String.(s.names)
    g["data"] = s.data
    s.err === nothing || (g["err"] = s.err)
    return g
end

function _write(g, nt::NamedTuple)
    a = HDF5.attrs(g)
    a["kind"] = "NamedTuple"
    a["keys"] = String[String(k) for k in keys(nt)]
    for (k, v) in pairs(nt)
        _write(g, String(k), v)
    end
    return g
end

function _read(g, k::AbstractString)
    haskey(g, k) || return nothing
    o = g[k]
    return o isa HDF5.Group ? _read(o) : read(o)
end

function _read(g)
    kind = HDF5.attrs(g)["kind"]
    kind == "Solution" && return _solution(g)
    kind == "NamedTuple" && return _namedtuple(g)
    error("load: unknown kind $kind")
end

function _solution(g)
    a = HDF5.attrs(g)
    B = BACKENDS[a["backend"]]
    b = B((convert(fieldtype(B, f), a[String(f)]) for f in fieldnames(B))...)
    return Solution(
        read(g["t"]),
        Symbol.(read(g["names"])),
        read(g["data"]),
        haskey(g, "err") ? read(g["err"]) : nothing,
        b,
    )
end

function _namedtuple(g)
    ks = Symbol.(HDF5.attrs(g)["keys"])
    return NamedTuple{Tuple(ks)}(Tuple(_read(g, String(k)) for k in ks))
end

"""
    save(path, x; group = "/") -> path

Write `x` to the HDF5 file `path`. `x` is a [`Solution`](@ref) or a `NamedTuple` whose
values are numbers, arrays, strings, `nothing`, `Solution`s, or
`NamedTuple`s.

Use [`load`](@ref) to reconstruct the `Solution`. `group = "/"` truncates the file. 
Any other `group` appends, replacing that group only, so one file can hold a parameter scan
(e.g. for d/λ = 0.2, 0.3, 0.4).
"""
function save(path::AbstractString, x; group::AbstractString = "/")
    HDF5.h5open(path, group == "/" ? "w" : "cw") do f
        group == "/" && return _write(f, x)
        haskey(f, group) && HDF5.delete_object(f, group)
        _write(HDF5.create_group(f, group), x)
    end
    return path
end

"""
    load(path; group = "/")

Inverse of [`save`](@ref). Returns the `Solution` or `NamedTuple` stored at `group`.
"""
load(path::AbstractString; group::AbstractString = "/") =
    HDF5.h5open(path, "r") do f
        _read(group == "/" ? f : f[group])
    end