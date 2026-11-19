"""
    LindbladProjMPO(𝓛::MPO)

The Lindbladian `𝓛`, an MPO on the vectorized sites of `vectorized_siteinds`, projected
onto a window of `rho` for the steady-state problem. The window covers the two sites
`ℓ:ℓ+1`, or the whole chain when it has one site. The left state is held fixed at the
vectorized identity `|1⟩`, which the site indices of `𝓛` determine; it is stored in `id`
as a product MPS and is never updated or truncated. `rho` is in the mixed canonical
gauge, so its blocks `L_ρ` and `R_ρ` outside the window are isometries.

Every cut lies between physical sites, so `|1⟩ = |1_L⟩ ⊗ |1_ℓ⟩ ⊗ |1_R⟩` factorizes across
both cuts of the window, and the bra side is the test map `P_{ℓ1} = L₁ ⊗ 1 ⊗ R₁` with

    L₁ = L_ρ + (1 - L_ρ L_ρ†) |1_L⟩⟨1_L|L_ρ / ⟨1_L|L_ρ L_ρ†|1_L⟩

and `R₁` its mirror image. Then `L₁† L_ρ = 1` and `L₁ L_ρ† |1_L⟩ = |1_L⟩`, so
`P_{ℓ1}† P_{ℓρ} = 1` and `P_{ℓ1} |φ₁⟩ = |1⟩` with `|φ₁⟩ = P_{ℓρ}† |1⟩`. Since `⟨1|𝓛 = 0`,
`⟨φ₁|` is an exact left null vector of the projected operator `P_{ℓ1}† 𝓛 P_{ℓρ}`, and
`⟨φ₁|phi⟩ = ⟨1|rho⟩` for the window tensor `phi` of `rho`. The denominator is
`‖⟨1_L|L_ρ‖²`, nonzero whenever `⟨1|rho⟩ ≠ 0`.

`env`, `idenv` and `trenv` cache the environments of `⟨rho|𝓛|rho⟩`, `⟨1|𝓛|rho⟩` and
`⟨1|rho⟩`, from which `position!` assembles `envL = L₁†(·)L_ρ` and `envR = R₁†(·)R_ρ`.
`bra1` is `⟨φ₁|`, and `cosine` is `‖φ₁‖/‖1‖ = 1/‖P_{ℓ1}‖`, the cosine between `|1⟩` and
the range of `P_{ℓρ}`. `id` has its own link indices, so they cannot collide with those
of `rho`.
"""
mutable struct LindbladProjMPO
    𝓛::MPO
    id::MPS
    idnorm::Float64
    env::EnvCache
    idenv::EnvCache
    trenv::EnvCache
    envL::ITensor
    envR::ITensor
    bra1::ITensor
    cosine::Float64
end

function LindbladProjMPO(𝓛::MPO)
    any(hasqns, 𝓛) && throw(ArgumentError("LindbladProjMPO is implemented for dense indices only"))
    id = vectorized_identity(firstsiteinds(𝓛; plev = 0))
    N = length(𝓛)
    return LindbladProjMPO(
        𝓛, id, _norm(id), EnvCache(N), EnvCache(N), EnvCache(N),
        ITensor(), ITensor(), ITensor(), 1.0,
    )
end

Base.length(P::LindbladProjMPO) = length(P.𝓛)

(P::LindbladProjMPO)(v::ITensor) = ITensors.product(P, v)

# Sites in a window: two, or the one site of a one-site chain.
_windowsize(P::LindbladProjMPO) = min(2, length(P))

# Environment of the test map at one cut of the window, from the environments `A`, `B`
# and `C` of ⟨rho|𝓛|rho⟩, ⟨1|𝓛|rho⟩ and ⟨1|rho⟩ on that side, where `λ` and `μ` are the
# links of `rho` and `id` at the cut. Every link of `id` has dimension 1, so
# `C = ⟨1_L|L_ρ` is a row vector, and
#
#     L₁†(·)L_ρ = L_ρ†(·)L_ρ + C† (⟨1_L|(·)L_ρ - C L_ρ†(·)L_ρ) / ‖C‖².
#
# The same function gives R₁†(·)R_ρ from the right environments.
function _testenv(A::ITensor, B::ITensor, C::ITensor, λ::Index, μ::Index)
    nC = norm(C)
    iszero(nC) && error("⟨1_L|L_ρ = 0, so rho is traceless")
    Cdag = replaceinds(dag(C), (λ, μ), (λ', μ'))
    return A + Cdag * (B - replaceinds(C, (λ, μ), (λ', μ')) * A) / nC^2
end

function ITensorMPS.position!(P::LindbladProjMPO, rho::MPS, ℓ::Int)
    n = _windowsize(P)
    reposition!(P.env, P.𝓛, rho, rho, ℓ; nsite = n)
    reposition!(P.idenv, P.𝓛, P.id, rho, ℓ; nsite = n)
    reposition!(P.trenv, nothing, P.id, rho, ℓ; nsite = n)
    r = ℓ + n - 1
    P.envL = ℓ == 1 ? lenv(P.env) : _testenv(
        lenv(P.env), lenv(P.idenv), lenv(P.trenv), linkind(rho, ℓ - 1), linkind(P.id, ℓ - 1)
    )
    P.envR = r == length(P) ? renv(P.env) : _testenv(
        renv(P.env), renv(P.idenv), renv(P.trenv), linkind(rho, r), linkind(P.id, r)
    )
    P.bra1 = _projbra(P.trenv, P.id)
    P.cosine = norm(P.bra1) / P.idnorm
    return P
end

function ITensors.product(P::LindbladProjMPO, v::ITensor)
    Pv = v * P.envL
    for k in site_range(P.env)
        Pv *= P.𝓛[k]
    end
    return checkinds(noprime(Pv * P.envR), v)
end