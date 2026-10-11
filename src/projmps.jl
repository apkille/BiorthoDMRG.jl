"""
    BiorthoProjMPS(Lref::MPS, Rref::MPS)

The rank-one operator `|Rref⟩⟨Lref|` projected onto the open window, with `chi`
as the bra and `psi` as the ket. It is the deflation term for a converged pair
`(Lref, Rref)`; `BiorthoProjMPO_MPS` applies the weight.

`Lref` and `Rref` are the reference states. `cacheL` caches the environments of
`⟨Lref|psi⟩` and `cacheR` those of `⟨Rref|chi⟩`. `braL` is `⟨Lref|` restricted
to the window through `psi`'s environments, and `braR` is `⟨Rref|` restricted
through `chi`'s; `position!` recomputes both.

`BiorthoProjMPS(Lref, Rref)` rescales `Rref` so that `⟨Lref|Rref⟩ = 1`, gives
both references new link indices so they cannot collide with those of `chi` or
`psi`, and creates empty caches.
"""
mutable struct BiorthoProjMPS <: AbstractBiorthoProjMPO
    Lref::MPS
    Rref::MPS
    cacheL::EnvCache
    cacheR::EnvCache
    braL::ITensor
    braR::ITensor
end

function BiorthoProjMPS(Lref::MPS, Rref::MPS)
    length(Lref) == length(Rref) ||
        throw(DimensionMismatch("left and right references have different lengths"))
    overlap = inner(Lref, Rref)
    iszero(overlap) && throw(ArgumentError("references are orthogonal: ⟨Lref|Rref⟩ = 0"))
    N = length(Lref)
    return BiorthoProjMPS(
        _simlinks(Lref), _simlinks(inv(overlap) * Rref), EnvCache(N), EnvCache(N), ITensor(), ITensor()
    )
end

# Copy of `M` with every link index replaced by a new one.
function _simlinks(M::MPS)
    M = copy(M)
    for k in 1:(length(M) - 1)
        l = commonind(M[k], M[k + 1])
        l̃ = sim(l)
        M[k] = replaceind(M[k], l, l̃)
        M[k + 1] = replaceind(M[k + 1], l, l̃)
    end
    return M
end

# ⟨ref| restricted to the window of `E`, with conjugated entries, so that
# contracting it with a window tensor gives the overlap.
function _projbra(E::EnvCache, ref::MPS)
    bra = lenv(E)
    for k in site_range(E)
        bra *= dag(ref[k])
    end
    return bra * renv(E)
end

function ITensorMPS.position!(P::BiorthoProjMPS, chi::MPS, psi::MPS, ℓ::Int)
    reposition!(P.cacheL, nothing, P.Lref, psi, ℓ)
    reposition!(P.cacheR, nothing, P.Rref, chi, ℓ)
    P.braL = _projbra(P.cacheL, P.Lref)
    P.braR = _projbra(P.cacheR, P.Rref)
    return P
end

function ITensors.product(P::BiorthoProjMPS, v::ITensor)
    return checkinds(scalar(P.braL * v) * dag(P.braR), v)
end

function adjoint_product(P::BiorthoProjMPS, v::ITensor)
    return checkinds(scalar(P.braR * v) * dag(P.braL), v)
end

Base.length(P::BiorthoProjMPS) = length(P.Lref)