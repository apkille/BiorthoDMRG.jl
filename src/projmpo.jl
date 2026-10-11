"""
    AbstractBiorthoProjMPO

Supertype for operators projected onto the open window of a biorthogonal pair
`(chi, psi)`. Subtypes implement `ITensorMPS.position!(P, chi, psi, ℓ)` to move
the window to sites `ℓ:ℓ+1`, `ITensors.product(P, v)` to apply the projected
operator to `v`, and `adjoint_product(P, v)` to apply its adjoint. `P(v)` is
the same as `product(P, v)`.
"""
abstract type AbstractBiorthoProjMPO end

(P::AbstractBiorthoProjMPO)(v::ITensor) = ITensors.product(P, v)

"""
    adjoint_product(P::AbstractBiorthoProjMPO, v::ITensor)

Apply the adjoint of the projected operator `P` to `v`, using the same
environments as `product`.
"""
function adjoint_product end

# The output must live in the same space as `v`; a mismatch means the window does
# not match `v` or a cached environment is stale.
function checkinds(w::ITensor, v::ITensor)
    hassameinds(w, v) || error(
        "projected product does not return the indices of its input: the window " *
        "does not match v, or an environment is stale.",
    )
    return w
end

"""
    BiorthoProjMPO(H::MPO)

`H` projected onto the open window, with `chi` as the bra and `psi` as the ket.
`H` is the full MPO; `env` caches the left and right environments of
`⟨chi|H|psi⟩` outside the window. `BiorthoProjMPO(H)` creates an empty cache,
and the first `position!` fills it.
"""
struct BiorthoProjMPO <: AbstractBiorthoProjMPO
    H::MPO
    env::EnvCache
end

BiorthoProjMPO(H::MPO) = BiorthoProjMPO(H, EnvCache(length(H)))

Base.length(P::BiorthoProjMPO) = length(P.H)

function ITensorMPS.position!(P::BiorthoProjMPO, chi::MPS, psi::MPS, ℓ::Int)
    reposition!(P.env, P.H, chi, psi, ℓ)
    return P
end

function ITensors.product(P::BiorthoProjMPO, v::ITensor)
    Kv = v * lenv(P.env)
    for k in site_range(P.env)
        Kv *= P.H[k]
    end
    return checkinds(noprime(Kv * renv(P.env)), v)
end

function adjoint_product(P::BiorthoProjMPO, v::ITensor)
    Kv = prime(dag(v)) * lenv(P.env)
    for k in site_range(P.env)
        Kv *= P.H[k]
    end
    return checkinds(dag(Kv * renv(P.env)), v)
end