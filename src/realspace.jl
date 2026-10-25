"""
    MPIRealSpace(term, comm; parts = nothing, pinv_rtol = sqrt(eps(Float64)))

The chain of `term` split into contiguous parts over the ranks of `comm`, for real-space
parallel `biorthodmrg`. Rank `p` sweeps the sites `parts[p + 1]`, and neighboring ranks
update the bond between their parts together. `parts` defaults to one part per rank, with
lengths that differ by at most one; every part needs at least two sites.

`term` must be identical on every rank, down to the index ids of its MPOs and reference
states, because the ranks exchange environments: build it from objects broadcast by rank 0.
It may be a `BiorthoProjMPO`, a `BiorthoProjMPS`, or a `BiorthoProjSum` of these.

Each rank holds its own representation of the pair, and the bond matrices at the bond
between two parts. The initial guess at that bond joins the two centers with the
pseudo-inverses of the bond matrices, dropping singular values at or below `pinv_rtol`
times the largest. The join is exact when neither rank has changed the states since the
last update of the bond.

Call `biorthodmrg(R, chi0, psi0; kwargs...)` on every rank, with the keywords of
`biorthodmrg` except `observer`. `chi0` and `psi0` are read on rank 0 only. It returns
`(E, chi, psi)` on rank 0, in biorthonormal form centered on site 1, and `nothing`
elsewhere.
"""
struct MPIRealSpace{T <: AbstractBiorthoProjMPO, C}
    term::T
    comm::C
    parts::Vector{UnitRange{Int}}
    pinv_rtol::Float64
end

Base.length(R::MPIRealSpace) = length(R.term)

biorthodmrg(R::MPIRealSpace, psi0::MPS, sweeps::Sweeps; kwargs...) =
    biorthodmrg(R, psi0, psi0, sweeps; kwargs...)

# `n` contiguous parts of `1:N` whose lengths differ by at most one.
_equalparts(N::Int, n::Int) = [(div(p * N, n) + 1):div((p + 1) * N, n) for p in 0:(n - 1)]

# The environment caches of `P`, each with its operator, bra, and ket, in a fixed order,
# so that ranks holding the same `P` exchange environments cache by cache.
_envspecs(P::BiorthoProjMPO, chi::MPS, psi::MPS) = [(P.env, P.H, chi, psi)]
_envspecs(P::BiorthoProjMPS, chi::MPS, psi::MPS) =
    [(P.cacheL, nothing, P.Lref, psi), (P.cacheR, nothing, P.Rref, chi)]
_envspecs(P::BiorthoProjSum, chi::MPS, psi::MPS) =
    mapreduce(t -> _envspecs(t, chi, psi), vcat, P.terms)
_envspecs(P::AbstractBiorthoProjMPO, ::MPS, ::MPS) =
    throw(ArgumentError("real-space sweeps are not implemented for $(typeof(P))"))

# Left environments of sites 1:k, one transfer beyond the cached ones of sites 1:k-1,
# returned without being stored.
function _extendleft(P::AbstractBiorthoProjMPO, chi::MPS, psi::MPS, k::Int)
    envs = map(_envspecs(P, chi, psi)) do (E, op, bra, ket)
        E.lpos == k - 1 || error("the left environments end at site $(E.lpos), not $(k - 1)")
        return transfer(lenv(E), ket[k], siteop(op, k), bra[k])
    end
    return convert(Vector{ITensor}, envs)
end

# Right environments of sites k:N, one transfer beyond the cached ones of sites k+1:N,
# returned without being stored.
function _extendright(P::AbstractBiorthoProjMPO, chi::MPS, psi::MPS, k::Int)
    envs = map(_envspecs(P, chi, psi)) do (E, op, bra, ket)
        E.rpos == k + 1 || error("the right environments start at site $(E.rpos), not $(k + 1)")
        return transfer(renv(E), ket[k], siteop(op, k), bra[k])
    end
    return convert(Vector{ITensor}, envs)
end

# Store `envs`, received from the neighboring rank, as the left environments of sites 1:k.
function _setleft!(P::AbstractBiorthoProjMPO, chi::MPS, psi::MPS, k::Int, envs::Vector{ITensor})
    specs = _envspecs(P, chi, psi)
    length(specs) == length(envs) || throw(DimensionMismatch("one environment per cache"))
    for (spec, X) in zip(specs, envs)
        E = first(spec)
        k ≥ 1 && (E.LR[k] = X)
        E.lpos = k
    end
    return P
end

# Store `envs`, received from the neighboring rank, as the right environments of sites k:N.
function _setright!(P::AbstractBiorthoProjMPO, chi::MPS, psi::MPS, k::Int, envs::Vector{ITensor})
    specs = _envspecs(P, chi, psi)
    length(specs) == length(envs) || throw(DimensionMismatch("one environment per cache"))
    for (spec, X) in zip(specs, envs)
        E = first(spec)
        k ≤ length(E) && (E.LR[k] = X)
        E.rpos = k
    end
    return P
end

# Move the center of the biorthonormal form from site `j` to site `k ≥ j`, refactorizing
# each bond with `biorthoreplacebond!` and `kwargs`.
function _moveright!(chi::MPS, psi::MPS, j::Int, k::Int; cutoff::Real = eps(Float64), kwargs...)
    for ℓ in j:(k - 1)
        biorthoreplacebond!(
            chi, psi, ℓ, chi[ℓ] * chi[ℓ + 1], psi[ℓ] * psi[ℓ + 1]; ortho = "left", cutoff, kwargs...
        )
    end
    return chi, psi
end

"""
    biorthosplitbond!(chi, psi, ℓ, phi_chi, phi_psi; basis = :biorthonormal, biortho_cutoff = 0, svd_kwargs...)

Split the two-site blocks `phi_chi` and `phi_psi` at the bond between sites `ℓ` and
`ℓ+1` into a left basis, a bond matrix, and a right basis for each state,
`phi_psi ≈ Xpsi * Cpsi * Ypsi` and `phi_chi ≈ Xchi * Cchi * Ychi`.

The left bases are those of `biorthoreplacebond!` with `ortho = "left"`. The right bases
come from the same factorization, with the same keywords, applied over site `ℓ+1` and
its right link to what that leaves on site `ℓ+1`, so that contracting `dag(Ychi)` with
`Ypsi` over these indices gives the identity. Site `ℓ` receives the left bases and site
`ℓ+1` the right bases. Returns `Cchi`, `Cpsi`, and the information of
`biorthoreplacebond!` with the larger truncation error of the two factorizations and the
smaller `sigma_min`.
"""
function biorthosplitbond!(
        chi::MPS, psi::MPS, ℓ::Int, phi_chi::ITensor, phi_psi::ITensor;
        basis::Symbol = :biorthonormal, biortho_cutoff::Real = 0, svd_kwargs...,
    )
    left = biorthoreplacebond!(
        chi, psi, ℓ, phi_chi, phi_psi; ortho = "left", basis, biortho_cutoff, svd_kwargs...
    )
    is, tags = uniqueinds(psi[ℓ + 1], psi[ℓ]), "Link,l=$ℓ"
    Ypsi, Ychi, right = basis === :shared ?
        _shared_basis(psi[ℓ + 1], chi[ℓ + 1], is; tags, svd_kwargs...) :
        _biortho_basis(psi[ℓ + 1], chi[ℓ + 1], is; biortho_cutoff, tags, svd_kwargs...)
    Cchi, Cpsi = chi[ℓ + 1] * dag(Ypsi), psi[ℓ + 1] * dag(Ychi)
    chi[ℓ + 1], psi[ℓ + 1] = Ychi, Ypsi
    spec(a, b) = ITensors.Spectrum(a.eigs, max(a.truncerr, b.truncerr))
    return Cchi, Cpsi, (;
        spec_psi = spec(left.spec_psi, right.spec_psi),
        spec_chi = spec(left.spec_chi, right.spec_chi),
        sigma_min = min(left.sigma_min, right.sigma_min),
    )
end

# Moore–Penrose pseudo-inverse of a bond matrix, with its two indices exchanged. Singular
# values at or below `rtol` times the largest are dropped.
function _pinv(C::ITensor; rtol::Real)
    i, j = inds(C)
    return itensor(LinearAlgebra.pinv(array(C, i, j); rtol), j, i)
end

_bondinverses(Cchi::ITensor, Cpsi::ITensor, rtol::Real) =
    (; chi = _pinv(Cchi; rtol), psi = _pinv(Cpsi; rtol))

# The site tensors of all parts in rank order, with the pseudo-inverses of the bond
# matrices absorbed into the last site of every part but the last.
function _glue(pieces)
    chis, psis = ITensor[], ITensor[]
    for (chik, psik, Cinv) in pieces
        append!(chis, chik)
        append!(psis, psik)
        if !isnothing(Cinv)
            chis[end] *= Cinv.chi
            psis[end] *= Cinv.psi
        end
    end
    return MPS(chis), MPS(psis)
end

_merge_stats(s::SweepStats, t::SweepStats) = SweepStats(
    max(s.maxtruncerr, t.maxtruncerr), min(s.sigma_min, t.sigma_min),
    min(s.cosangle_min, t.cosangle_min), max(s.maxresidual, t.maxresidual),
)

# Largest link dimension within `part` and at its right boundary.
_maxlinkdim(psi::MPS, part::UnitRange{Int}) =
    maximum(k -> dim(commonind(psi[k], psi[k + 1])), first(part):min(last(part), length(psi) - 1))

# `reports` holds `(energy, stats, maxlinkdim)` of every rank; `spread` is the largest
# deviation of a rank's energy from rank 0's, which vanishes at the fixed point.
function _print_realspace_sweep(sw::Int, reports, elapsed::Real)
    energies = first.(reports)
    s = reduce(_merge_stats, getindex.(reports, 2))
    @printf(
        "After sweep %d energy=%s spread=%.2E maxlinkdim=%d maxerr=%.2E sigma_min=%.2E cosangle_min=%.2E time=%.3f\n",
        sw, first(energies), maximum(E -> abs(E - first(energies)), energies),
        maximum(last, reports), s.maxtruncerr, s.sigma_min, s.cosangle_min, elapsed
    )
    return nothing
end