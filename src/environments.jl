"""
    EnvCache(N)

Left and right environments of an `N`-site chain around an open window.

`LR[k]` holds the left environment of sites `1:k` for `k ≤ lpos` and the right
environment of sites `k:N` for `k ≥ rpos`; sites `lpos+1:rpos-1` form the window.
Each environment contracts the ket, the operator, and the conjugated bra over its
sites. The operator is an MPO, or the identity when it is `nothing`.
"""
mutable struct EnvCache
    lpos::Int
    rpos::Int
    LR::Vector{ITensor}
end

EnvCache(N::Int) = EnvCache(0, N + 1, Vector{ITensor}(undef, N))

Base.length(E::EnvCache) = length(E.LR)
site_range(E::EnvCache) = (E.lpos + 1):(E.rpos - 1)
lenv(E::EnvCache) = E.lpos ≥ 1 ? E.LR[E.lpos] : ITensor(true)
renv(E::EnvCache) = E.rpos ≤ length(E) ? E.LR[E.rpos] : ITensor(true)

siteop(H::MPO, k::Int) = H[k]
siteop(::Nothing, ::Int) = nothing

# operator sandwich, where bra and ket share link indices, so the bra is primed
transfer(X::ITensor, ket::ITensor, op::ITensor, bra::ITensor) = X * ket * op * dag(prime(bra))
# overlap, where bra and ket carry distinct link indices, so only site indices contract
transfer(X::ITensor, ket::ITensor, ::Nothing, bra::ITensor) = X * ket * dag(bra)

"""
    reposition!(E, op, bra, ket, ℓ; nsite=2)

Move the window to sites `ℓ:ℓ+nsite-1`. 

Cached environments are reused and only missing ones are computed, so `bra` and `ket` must be unchanged at every site
covered by a reused environment.
"""
function reposition!(E::EnvCache, op, bra::MPS, ket::MPS, ℓ::Int; nsite::Int = 2)
    1 ≤ ℓ ≤ length(E) - nsite + 1 || throw(BoundsError(E.LR, ℓ))
    while E.lpos < ℓ - 1
        k = E.lpos + 1
        E.LR[k] = transfer(lenv(E), ket[k], siteop(op, k), bra[k])
        E.lpos = k
    end
    E.lpos = ℓ - 1
    while E.rpos > ℓ + nsite
        k = E.rpos - 1
        E.LR[k] = transfer(renv(E), ket[k], siteop(op, k), bra[k])
        E.rpos = k
    end
    E.rpos = ℓ + nsite
    return E
end