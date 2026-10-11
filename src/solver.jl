# Index of the right Ritz pair: first in `which` order, or the one overlapping
# most with the current left block.
function _select_right(vecsR, phi_chi, target::Symbol)
    target in (:which, :overlap) ||
        throw(ArgumentError("target must be :which or :overlap, got :$target"))
    target === :which && return 1
    return argmax(i -> abs(scalar(dag(phi_chi) * vecsR[i])) / norm(vecsR[i]), eachindex(vecsR))
end

# Selector for the adjoint problem, whose relevant eigenvalue is conj(E). In
# complex arithmetic it targets conj(E) directly. In real arithmetic KrylovKit
# keeps conjugate pairs together and accepts only conjugation-symmetric
# selectors, so `which` is reused.
function _adjoint_which(which::Symbol, E::Number, phi_chi::ITensor)
    eltype(phi_chi) <: Real && return which
    return EigSorter(λ -> abs(λ - conj(E)))
end

"""
    sssolver(P, phi_psi, phi_chi; tol, krylovdim, maxiter, verbosity)

Local solver for the steady state of a Lindbladian, used as
`biorthodmrg(𝓛, id; solver = sssolver)` with `id = vectorized_identity(d)`.

`phi_chi` is the local block of the vectorized identity, which is an exact left
null vector of `P`, so the steady state is a right null vector and there is no
eigenvalue to select. The solver writes it as `x = x₀ + y`, with `x₀ = phi_psi`
rescaled to unit trace and `y` orthogonal to `phi_chi`, and solves `P y = -P x₀`
with GMRES inside that orthogonal complement, so the trace stays fixed.
`phi_chi` is returned unchanged and `x` is normalized so that `⟨phi_chi|x⟩ = 1`.
The returned value is the Rayleigh quotient of `x`, zero at convergence, and
`info.residual` is `‖P x‖ / ‖x‖`.
"""
function sssolver(
        P::AbstractBiorthoProjMPO, phi_psi::ITensor, phi_chi::ITensor;
        tol::Real = 1.0e-14, krylovdim::Int = 30, maxiter::Int = 10, verbosity::Int = 0, kwargs...,
    )
    û = phi_chi / norm(phi_chi)
    project(v) = v - û * scalar(dag(û) * v)
    x0 = phi_psi / scalar(dag(û) * phi_psi)
    y, _ = linsolve(v -> project(P(project(v))), -project(P(x0)), 0 * x0; tol, krylovdim, maxiter, verbosity)
    x = x0 + project(y)
    x /= scalar(dag(phi_chi) * x)
    Px = P(x)
    λ = scalar(dag(x) * Px) / scalar(dag(x) * x)
    return λ, x, phi_chi, (; cosangle = 1 / (norm(phi_chi) * norm(x)), residual = norm(Px) / norm(x))
end

"""
    biorthoproblemsolver(P, phi_psi, phi_chi; which=:SR, target=:which, howmany, tol, krylovdim, maxiter, verbosity)

Solve the local eigenproblem of the projected operator `P` from both sides. Arnoldi
on `P`, starting from `phi_psi`, gives `E` and the right eigenvector; Arnoldi on
the adjoint, starting from `phi_chi` and targeted at `conj(E)`, gives the left
eigenvector.

`target=:which` takes the first right pair in KrylovKit's `which` order.
`target=:overlap` takes the right pair, among `howmany`, that overlaps most with
the current left block. `tol`, `krylovdim`, `maxiter` and `verbosity` are passed
to `KrylovKit.eigsolve`.

Returns `(E, phi_psi, phi_chi, info)`, normalized so that `⟨phi_chi|phi_psi⟩ = 1`
with equal norms. `info.cosangle` is the cosine of the angle between the two
vectors before normalization, which vanishes at an exceptional point of `P`.
`info.residual` is the larger of the relative residuals of the two vectors.
"""
function biorthoproblemsolver(
        P::AbstractBiorthoProjMPO, phi_psi::ITensor, phi_chi::ITensor;
        which::Symbol = :SR, target::Symbol = :which, howmany::Int = target === :overlap ? 2 : 1,
        tol::Real = 1.0e-14, krylovdim::Int = 3, maxiter::Int = 1, verbosity::Int = 0,
    )
    kw = (; ishermitian = false, tol, krylovdim, maxiter, verbosity)
    valsR, vecsR, infoR = eigsolve(P, phi_psi, howmany, which; kw...)
    i = _select_right(vecsR, phi_chi, target)
    energy, rvec = valsR[i], vecsR[i]
    valsL, vecsL, infoL = eigsolve(
        v -> adjoint_product(P, v), phi_chi, 1, _adjoint_which(which, energy, phi_chi); kw...
    )
    j = argmin(abs.(valsL .- conj(energy)))
    lvec = vecsL[j]
    overlap = scalar(dag(lvec) * rvec)
    iszero(overlap) && error("local left and right eigenvectors are orthogonal (exceptional point of P)")
    cosangle = abs(overlap) / (norm(rvec) * norm(lvec))
    residual = max(infoR.normres[i] / norm(rvec), infoL.normres[j] / norm(lvec))
    lvec /= conj(overlap)
    a = sqrt(norm(lvec) / norm(rvec))
    return energy, a * rvec, lvec / a, (; cosangle, residual)
end