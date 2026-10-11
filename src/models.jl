"""
    Models

Model Hamiltonians as MPOs, for examples and tests. All chains are open.
"""
module Models

using ITensors: ITensor, Index, addtags, commonind, dag, delta, dense, dim, hascommoninds, inds,
    onehot, op, prime, replaceind, replaceinds, replaceprime, sim, swapprime
using ITensorMPS: MPO, MPS, OpSum

export doubled_siteinds, ising_imaginary_field, lindbladian, nonreciprocal_ssh,
    vectorized_identity, xxz

include("lindblad.jl")

"""
    xxz(sites; Δ, g = 0.0, J = 1.0, h = 0.0)

XXZ chain with non-reciprocal spin flips. Each bond carries
`J/2 (e^g S⁺_{j+1} S⁻_j + e^-g S⁻_{j+1} S⁺_j) + Δ Sᶻ_j Sᶻ_{j+1}` and each site
`-h Sᶻ_j`. For any `g` it equals `S H₀ S⁻¹` with `H₀` the Hermitian chain at
`g = 0` and `S = exp(g Σ_j j Sᶻ_j)`, so the spectra agree, and the right and left
eigenvectors are `S|ψ₀⟩` and `S⁻¹|ψ₀⟩`. Under Jordan–Wigner it is the interacting
Hatano–Nelson model. Use `"S=1/2"` sites.
"""
function xxz(sites; Δ::Real, g::Real = 0.0, J::Real = 1.0, h::Real = 0.0)
    os = OpSum()
    for j in 1:(length(sites) - 1)
        os += J * exp(g) / 2, "S+", j + 1, "S-", j
        os += J * exp(-g) / 2, "S-", j + 1, "S+", j
        iszero(Δ) || (os += (Δ, "Sz", j, "Sz", j + 1))
    end
    if !iszero(h)
        for j in eachindex(sites)
            os += -h, "Sz", j
        end
    end
    return MPO(os, sites)
end

"""
    nonreciprocal_ssh(sites; t1, t2, γ, V = 0.0)

SSH chain of spinless fermions with non-reciprocal intracell hopping, sites
ordered `A₁, B₁, A₂, B₂, …`. Cell `j` carries
`(t1 + γ/2) c†_{A_j} c_{B_j} + (t1 - γ/2) c†_{B_j} c_{A_j}`, neighboring cells are
joined by `t2 (c†_{B_j} c_{A_{j+1}} + h.c.)`, and `V` adds `V n_x n_{x+1}` on every
bond. For `|t1| > |γ|/2` it is similar to the Hermitian chain with intracell
hopping `sign(t1) * sqrt(t1^2 - γ^2/4)`, so the spectra agree. Use `"Fermion"`
sites; the number of sites must be even.
"""
function nonreciprocal_ssh(sites; t1::Real, t2::Real, γ::Real, V::Real = 0.0)
    N = length(sites)
    iseven(N) || throw(ArgumentError("the SSH chain needs an even number of sites"))
    os = OpSum()
    for a in 1:2:N                  # a = A_j, a + 1 = B_j
        os += t1 + γ / 2, "Cdag", a, "C", a + 1
        os += t1 - γ / 2, "Cdag", a + 1, "C", a
    end
    for b in 2:2:(N - 2)            # b = B_j, b + 1 = A_{j+1}
        os += t2, "Cdag", b, "C", b + 1
        os += t2, "Cdag", b + 1, "C", b
    end
    if !iszero(V)
        for x in 1:(N - 1)
            os += V, "N", x, "N", x + 1
        end
    end
    return MPO(os, sites)
end

"""
    ising_imaginary_field(sites; J = 1.0, hx, hz)

Transverse-field Ising chain with an imaginary longitudinal field. Each bond
carries `-J Z_j Z_{j+1}` and each site `-hx X_j - i hz Z_j`, with Pauli `X`, `Z`.
The matrix is complex symmetric, so each left eigenvector is the complex
conjugate of the corresponding right eigenvector. At `hz = 0` it is Hermitian.
Use `"S=1/2"` sites.
"""
function ising_imaginary_field(sites; J::Real = 1.0, hx::Real, hz::Real)
    os = OpSum()
    for j in 1:(length(sites) - 1)
        os += -J, "Z", j, "Z", j + 1
    end
    for j in eachindex(sites)
        os += -hx, "X", j
        iszero(hz) || (os += (-im * hz, "Z", j))
    end
    return MPO(os, sites)
end

end