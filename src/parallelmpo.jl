"""
    SiteTracker()

References to the site tensors of `chi` and `psi` at the last call of `changedsites!`.
"""
mutable struct SiteTracker
    chi::Vector{ITensor}
    psi::Vector{ITensor}
end

SiteTracker() = SiteTracker(ITensor[], ITensor[])

# Sites at which `chi` or `psi` holds a different tensor object than at the last call,
# and every site on the first call or after a change of length.
function changedsites!(t::SiteTracker, chi::MPS, psi::MPS)
    N = length(psi)
    if length(t.psi) != N
        t.chi, t.psi = ITensor[chi[k] for k in 1:N], ITensor[psi[k] for k in 1:N]
        return collect(1:N)
    end
    ks = [k for k in 1:N if chi[k] !== t.chi[k] || psi[k] !== t.psi[k]]
    for k in ks
        t.chi[k], t.psi[k] = chi[k], psi[k]
    end
    return ks
end

"""
    MPIBiorthoSum(term, comm; mode, checksync = false)

Projected operator whose terms are spread over the ranks of `comm`. Each rank holds one
part of the Hamiltonian, for example `term = BiorthoProjMPO(H_p)` with `H = Σ_p H_p`, and
builds the environments for that part only. Every rank must call the constructor.

With `mode = :root`, only rank 0 runs `biorthodmrg`, and the other ranks run `serve!`.
Rank 0 sends the other ranks every updated site tensor and every window vector, and
collects the sum of their products. Every decision of the eigensolver and the truncation
is therefore made on rank 0 alone, and the mode does not depend on reproducible
arithmetic. Rank 0 calls `stopservice!` when it is done, for example in a `finally`
block, which releases the other ranks.

With `mode = :replicated`, every rank runs `biorthodmrg` on the same inputs. The sum of
the products is returned to every rank, and each rank repeats the eigensolver and the
truncation on identical data. The ranks stay in step only if their arithmetic is bitwise
reproducible, which in practice means the same CPU type and a fixed number of BLAS
threads. With `checksync = true`, each `position!` compares the newly replaced site
tensors across ranks and throws if they differ. Wrap the observer in `MPIObserver` and
set `outputlevel = 0` on every rank except 0.

The terms may have `Float64` entries on some ranks and `ComplexF64` entries on others.
Each product is promoted to the common entry type before the sum is taken.
"""
struct MPIBiorthoSum{T <: AbstractBiorthoProjMPO, C, S <: Number} <: AbstractBiorthoProjMPO
    term::T
    comm::C
    scalartype::Type{S}
    mode::Symbol
    isroot::Bool
    checksync::Bool
    tracker::SiteTracker
end

Base.length(P::MPIBiorthoSum) = length(P.term)

"""
    serve!(P::MPIBiorthoSum)

Worker loop for `mode = :root`, run on every rank of `P.comm` except 0: apply this
rank's term on request from rank 0 until rank 0 calls `stopservice!(P)`. An error on a
worker aborts the communicator, since rank 0 would otherwise wait in the reduction.
"""
function serve! end

"""
    stopservice!(P::MPIBiorthoSum)

End `serve!` on every worker. Rank 0 calls it once `biorthodmrg` has returned or
thrown, for example in a `finally` block.
"""
function stopservice! end

"""
    MPIObserver(obs, comm)

Wrap `obs` for `mode = :replicated`. `obs` runs on rank 0 of `comm` only (pass
`nothing` elsewhere), and its stopping decision is broadcast so that every rank leaves
the sweep loop together.
"""
struct MPIObserver{O, C}
    obs::O
    comm::C
    isroot::Bool
end

ITensorMPS.measure!(o::MPIObserver; kwargs...) =
    o.isroot ? ITensorMPS.measure!(o.obs; kwargs...) : nothing

function _record_sweep!(o::MPIObserver, energy, chi, psi, stats; kwargs...)
    o.isroot && _record_sweep!(o.obs, energy, chi, psi, stats; kwargs...)
    return o
end

# Promoted entry type of the tensors a projected operator holds besides the running
# states. `MPIBiorthoSum` promotes it over the ranks.
_scalartype(::AbstractBiorthoProjMPO) = ComplexF64
_scalartype(M::Union{MPS, MPO}) = mapreduce(k -> eltype(M[k]), promote_type, 1:length(M))
_scalartype(P::BiorthoProjMPO) = _scalartype(P.H)
_scalartype(P::BiorthoProjMPS) = promote_type(_scalartype(P.Lref), _scalartype(P.Rref))
_scalartype(P::BiorthoProjSum) =
    mapreduce(_scalartype, promote_type, P.terms; init = eltype(P.coeffs))
_scalartype(P::MPIBiorthoSum) = P.scalartype