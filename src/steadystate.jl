"""
    steadystate(𝓛::MPO; nsweeps, maxdim, mindim, cutoff, kwargs...)
    steadystate(𝓛::MPO, rho0::MPS; ...)
    steadystate(𝓛::MPO, rho0::MPS, sweeps::Sweeps; ...)

Two-site DMRG for the steady state of the Lindbladian `𝓛`, an MPO on doubled sites
ordered as by `doubled_siteinds`. Returns `(λ, rho)`: `rho` is the right null vector of
`𝓛` with unit trace, and `λ` its Rayleigh quotient, which vanishes at convergence, both
up to truncation.

The left state stays fixed at the vectorized identity, which `LindbladProjMPO` builds
from the site indices of `𝓛`, and only `rho` is swept, in the mixed canonical gauge. At
each bond `LindbladProjMPO` projects `𝓛` and gives the trace functional `⟨φ₁|`, and
`sssolver` removes the traceless part of the window tensor. `rho0` is any state on the
sites of `𝓛` with nonzero trace; without it, the sweeps start from the maximally mixed
state. The last form takes a `Sweeps` object; the others build it from the keywords.

`observer` is an ITensorMPS observer or a `BiorthoObserver`, which records the
vectorized identity as the left state. `outputlevel` is as in `ITensorMPS.dmrg`, and the
`linsolve_*` keywords are passed to `sssolver`. `noise` must be zero and indices must be
dense.

Between the ket and bra copies of a site, the identity has Schmidt rank equal to the
local dimension, and `LindbladProjMPO` needs its overlap with the kept space of `rho` to
have full rank there. If the steady state is locally pure, set `mindim` to at least the
local dimension.
"""
function steadystate(
        𝓛::MPO, rho0::MPS, sweeps::Sweeps;
        observer = ITensorMPS.NoObserver(), outputlevel::Int = 1, linsolve_tol::Real = 1.0e-14,
        linsolve_krylovdim::Int = 30, linsolve_maxiter::Int = 10, linsolve_verbosity::Int = 0,
    )
    nsw = ITensorMPS.nsweep(sweeps)
    any(sw -> ITensorMPS.noise(sweeps, sw) > 0, 1:nsw) &&
        throw(ArgumentError("noise is not defined for the steady-state solver"))
    P = LindbladProjMPO(𝓛)
    N = length(rho0)
    length(P) == N ||
        throw(DimensionMismatch("the Lindbladian and the initial state have different lengths"))
    all(k -> siteind(P.id, k) == siteind(rho0, k), 1:N) ||
        throw(ArgumentError("rho0 must live on the site indices of the Lindbladian"))
    iszero(inner(P.id, rho0)) && throw(ArgumentError("rho0 is traceless"))
    rho = orthogonalize(rho0, 1)
    solver_kwargs = (;
        tol = linsolve_tol, krylovdim = linsolve_krylovdim, maxiter = linsolve_maxiter,
        verbosity = linsolve_verbosity,
    )
    λ = zero(ComplexF64)
    for sw in 1:nsw
        stats = SweepStats()
        elapsed = @elapsed for (ℓ, ha) in ITensorMPS.sweepnext(N)
            ITensorMPS.position!(P, rho, ℓ)
            λ, phi, solve_info = sssolver(P, rho[ℓ] * rho[ℓ + 1]; solver_kwargs...)
            spec = ITensorMPS.replacebond!(
                rho, ℓ, phi;
                ortho = ha == 1 ? "left" : "right", which_decomp = "svd",
                maxdim = ITensorMPS.maxdim(sweeps, sw), mindim = ITensorMPS.mindim(sweeps, sw),
                cutoff = ITensorMPS.cutoff(sweeps, sw),
            )
            stats = _merge_stats(stats, solve_info; truncerr = spec.truncerr, sigma_min = P.cosine)
            ITensorMPS.measure!(
                observer; energy = λ, psi = rho, projected_operator = P, bond = ℓ, sweep = sw,
                half_sweep = ha, spec, outputlevel, sweep_is_done = (ℓ == 1 && ha == 2),
            )
        end
        outputlevel ≥ 1 && _print_sweep(sw, λ, rho, stats, elapsed)
        _record_sweep!(observer, λ, P.id, rho, stats; outputlevel)
        ITensorMPS.checkdone!(observer; energy = λ, psi = rho, sweep = sw, outputlevel) && break
    end
    return λ, rho
end

function steadystate(𝓛::MPO, sweeps::Sweeps; kwargs...)
    return steadystate(𝓛, vectorized_identity(firstsiteinds(𝓛; plev = 0)), sweeps; kwargs...)
end

function steadystate(
        args...; nsweeps::Int, maxdim = ITensorMPS.default_maxdim(),
        mindim = ITensorMPS.default_mindim(), cutoff = ITensorMPS.default_cutoff(Float64),
        noise = ITensorMPS.default_noise(), kwargs...,
    )
    sweeps = ITensorMPS._dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
    return steadystate(args..., sweeps; kwargs...)
end