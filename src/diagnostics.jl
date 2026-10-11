"""
    adjoint_mpo(H::MPO)

H† as an MPO: conjugate every tensor and exchange the prime levels of its site
indices (tagged "Site").  `dag(H)` alone is the elementwise conjugate, not H†.
"""
adjoint_mpo(H::MPO) = MPO([swapprime(dag(H[k]), 0, 1; tags = "Site") for k in 1:length(H)])

# ⟨x|x⟩ is real in exact arithmetic; taking the real part directly avoids the
# warning `norm` issues for complex states that are not orthogonalized.
_norm(x::MPS) = sqrt(real(inner(x, x)))

function relresidual(A::MPO, x::MPS, λ::Number)
    r2 = inner(A, x, A, x) - 2 * real(conj(λ) * inner(x', A, x)) + abs2(λ) * inner(x, x)
    return sqrt(max(real(r2), 0.0)) / _norm(x)
end

"""
    residuals(H, chi, psi, E; Hdag=adjoint_mpo(H)) -> (r_psi, r_chi)

Relative residuals ‖H|ψ⟩ − E|ψ⟩‖/‖ψ‖ and ‖H†|χ⟩ − Ē|χ⟩‖/‖χ‖, from
‖A|x⟩ − λ|x⟩‖² = ⟨Ax|Ax⟩ − 2Re(λ̄⟨x|A|x⟩) + |λ|²⟨x|x⟩.  The expansion cancels
large terms, so the resolution is limited to about √eps relative to ‖A‖.
"""
function residuals(H::MPO, chi::MPS, psi::MPS, E::Number; Hdag::MPO = adjoint_mpo(H))
    return relresidual(H, psi, E), relresidual(Hdag, chi, conj(E))
end

"""
    condition_number(chi, psi)

κ = ‖χ‖‖ψ‖ / |⟨χ|ψ⟩| ≥ 1; it equals 1 iff |χ⟩ ∝ |ψ⟩.
"""
condition_number(chi::MPS, psi::MPS) = _norm(chi) * _norm(psi) / abs(inner(chi, psi))