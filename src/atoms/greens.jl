const MU0, CLIGHT = 1.0, 1.0
const K0 = (3π)^(1/3) # Γ₀ = k0³|℘|²/(3π) = 1 for a unit dipole
const LAMBDA = 2π / K0

"""
    greens_coupling(ri::Vector{<:Real}, rj::Vector{<:Real}, dipole::Vector{<:Number})

Compute `g^{ij} = J^{ij} - i Γ^{ij}/2` from the free-space dyadic Green's tensor.

Units: `μ0 = c = 1` and `k0 = K0`, so `Γ₀ = 1` for a unit dipole.
"""
function greens_coupling(
    ri::Vector{<:Real}, rj::Vector{<:Real}, dipole::Vector{<:Number}
)
    # positions
    rvec = ri .- rj
    r = LinearAlgebra.norm(rvec)
    if r == 0.0
        return -im/2.0 * LinearAlgebra.dot(dipole, dipole)
    end
    rhat = rvec ./ r
    # coeffs
    A = K0^2*r^2 + im*K0*r - 1
    B = -K0^2*r^2 - 3im*K0*r + 3
    # green's tensor scalar prefactor
    C = -MU0 * (K0*CLIGHT)^2 * exp(im*K0*r) / (4π * K0^2 * r^3)
    # contractions
    ρρ = LinearAlgebra.dot(dipole, dipole) # ρ* ⋅ ρ
    ρr = LinearAlgebra.dot(dipole, rhat) * LinearAlgebra.dot(rhat, dipole) # (ρ* ⋅ r̂)(r̂ ⋅ ρ)
    return C * (A*ρρ + B*ρr)
end
function interactions(pos::Vector{<:Vector{<:Real}}, dipole::Vector{<:Number})
    N = length(pos)
    coh = zeros(Float64, N, N)
    decay = zeros(Float64, N, N)
    @inbounds for i in 1:N
        @inbounds for j in 1:N
            g = greens_coupling(pos[i], pos[j], dipole)
            coh[i,j] = real(g)
            decay[i,j] = -2 * imag(g)
        end
    end
    return coh, decay
end