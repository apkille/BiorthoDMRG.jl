abstract type Backend end

# settings for `biorthodmrg`. The `eigsolve_*` fields configure the local solver:
# GMRES in `steadystate`, Arnoldi in `eigenstate`.
Base.@kwdef struct Biortho <: Backend
    nsweeps::Int = 20
    cutoff::Float64 = 1e-12
    mindim::Int = 1
    maxdim::Int = 256
    eigsolve_tol::Float64 = 1e-12
    eigsolve_krylovdim::Int = 30
    eigsolve_maxiter::Int = 10
    mpo_cutoff::Float64 = 1e-14 # compression of the ℒ MPO
    residual_tol::Float64 = 0.0 # in units of Γ₀; 0 runs every sweep
    verbose::Bool = false
end

_opts(b::Biortho) = (; 
    b.nsweeps, 
    b.cutoff, 
    b.mindim, 
    b.maxdim, 
    b.eigsolve_tol,
    b.eigsolve_krylovdim, 
    b.eigsolve_maxiter,
    outputlevel = Int(b.verbose)
)

abstract type Register end
struct KetRegister{I} <: Register
    sites::Vector{I}
end
struct DoubledRegister{I} <: Register
    sites::Vector{I}
    id::ITensorMPS.MPS # cached vectorized_identity
end

natoms(r::KetRegister) = length(r.sites)
natoms(r::DoubledRegister) = length(r.sites) ÷ 2

abstract type Observable end
struct Population <: Observable end
struct Emission   <: Observable end

name(::Population) = :population
name(::Emission)   = :emission

materialize(::Population, reg, c) = population(reg)
materialize(::Emission,   reg, c) = emission(reg, c.decay)

struct Coherence <: Observable
    i::Int
    label::Symbol
end
Coherence(i::Int; label = Symbol("sm_", i)) = Coherence(i, label)
name(o::Coherence) = o.label
materialize(o::Coherence, reg, c) = coherence(reg, o.i)

struct Density <: Observable
    i::Int
    label::Symbol
end
Density(i::Int; label = Symbol("n_", i)) = Density(i, label)
name(o::Density) = o.label
materialize(o::Density, reg, c) = density(reg, o.i)

struct Correlation <: Observable
    i::Int
    j::Int
    label::Symbol
end
Correlation(i::Int, j::Int; label = Symbol("C_", i, "_", j)) = Correlation(i, j, label)
name(o::Correlation) = o.label
materialize(o::Correlation, reg, c) = correlation(reg, o.i, o.j)

struct Pairing <: Observable
    i::Int
    j::Int
    label::Symbol
end
Pairing(i::Int, j::Int; label = Symbol("M_", i, "_", j)) = Pairing(i, j, label)
name(o::Pairing) = o.label
materialize(o::Pairing, reg, c) = pairing(reg, o.i, o.j)

struct Density2 <: Observable
    i::Int
    j::Int
    label::Symbol
end
Density2(i::Int, j::Int; label = Symbol("nn_", i, "_", j)) = Density2(i, j, label)
name(o::Density2) = o.label
materialize(o::Density2, reg, c) = density2(reg, o.i, o.j)

coherences(N::Int)   = Observable[Coherence(i) for i in 1:N]
densities(N::Int)    = Observable[Density(i) for i in 1:N]
correlations(N::Int) = Observable[Correlation(i, j) for i in 1:N for j in i:N]
pairings(N::Int)     = Observable[Pairing(i, j) for i in 1:N-1 for j in i+1:N]
densities2(N::Int)   = Observable[Density2(i, j) for i in 1:N for j in i:N]

struct Inversion2 <: Observable
    i::Int
    j::Int
    label::Symbol
end
Inversion2(i::Int, j::Int; label = Symbol("zz_", i, "_", j)) = Inversion2(i, j, label)
name(o::Inversion2) = o.label
materialize(o::Inversion2, reg, c) = inversion2(reg, o.i, o.j)

inversions2(N::Int) = Observable[Inversion2(i, j) for i in 1:N-1 for j in i+1:N]

struct Solution{B<:Backend}
    t::Vector{Float64}
    names::Vector{Symbol}
    data::Matrix{ComplexF64} # length(t) × length(names)
    err::Union{Nothing,Matrix{ComplexF64}}
    backend::B
end
Base.getindex(s::Solution, k::Symbol) = s.data[:, _col(s, k)]
sem(s::Solution, k::Symbol) = s.err === nothing ? nothing : s.err[:, _col(s, k)]

function _col(s::Solution, k::Symbol)
    i = findfirst(==(k), s.names)
    i === nothing && throw(KeyError(k))
    return i
end
Base.keys(s::Solution) = s.names
Base.haskey(s::Solution, k::Symbol) = k in s.names
Base.show(io::IO, ::MIME"text/plain", s::Solution) =
    print(io, "Solution($(s.backend), $(length(s.t)) times, $(join(s.names, ", ")))")