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
    sssolver(P::LindbladProjMPO, phi; tol, krylovdim, maxiter, verbosity)

Local solver of `steadystate`. With `⟨φ₁| = P.bra1` the trace functional on the window,
`phi` is rescaled to unit trace, `⟨φ₁|phi⟩ = 1`, and GMRES started from zero solves

    P δ = P phi,    ⟨φ₁|δ⟩ = 0.

Then `x = phi - δ` is the right null vector of `P` with unit trace, the result of one step
of inverse iteration with the shift at the known eigenvalue `0`. `⟨φ₁|` is an exact left
null vector of `P`, so `P` maps the orthogonal complement of `|φ₁⟩` into itself, and the
projections onto that complement remove only rounding errors. The solution is unique if
and only if the null space of `P` is one-dimensional and not orthogonal to `|φ₁⟩`.
`tol`, `krylovdim`, `maxiter` and `verbosity` are passed to `KrylovKit.linsolve`.

Returns `(λ, x, info)`. `λ` is the Rayleigh quotient of `x`, zero at convergence.
`info.cosangle` is the cosine between `x` and `|φ₁⟩`, which vanishes as the null vector of
`P` becomes orthogonal to `|φ₁⟩`, and `info.residual` is `‖P x‖ / ‖x‖`.
"""
function sssolver(
        P::LindbladProjMPO, phi::ITensor;
        tol::Real = 1.0e-14, krylovdim::Int = 30, maxiter::Int = 10, verbosity::Int = 0,
    )
    û = dag(P.bra1) / norm(P.bra1)
    project(v) = v - û * scalar(dag(û) * v)
    phi = phi / scalar(P.bra1 * phi)
    δ, _ = linsolve(
        v -> project(P(project(v))), project(P(phi)), 0 * phi; tol, krylovdim, maxiter, verbosity
    )
    x = phi - project(δ)
    Px = P(x)
    λ = scalar(dag(x) * Px) / scalar(dag(x) * x)
    return λ, x, (; cosangle = 1 / (norm(P.bra1) * norm(x)), residual = norm(Px) / norm(x))
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