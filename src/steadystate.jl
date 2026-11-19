# Starts `ℓ` of the windows of `n` sites, with their half sweeps: left to right, then
# right to left.
function _windowsweep(N::Int, n::Int)
    ℓs = 1:(N - n + 1)
    return Iterators.flatten((((ℓ, 1) for ℓ in ℓs), ((ℓ, 2) for ℓ in reverse(ℓs))))
end

# Split `phi`, the window tensor on the sites `ks` of `rho`, back into site tensors by
# successive SVDs truncated with `svd_kwargs`, starting from the side the sweep comes
# from. With `ortho = "left"` every site but the last is a left isometry, and with
# `ortho = "right"` every site but the first is a right isometry. Returns the spectrum of
# the bond with the largest truncation error.
function _replacewindow!(rho::MPS, ks::UnitRange{Int}, phi::ITensor; ortho::String, svd_kwargs...)
    order = ortho == "left" ? ks : ortho == "right" ? reverse(ks) :
        throw(ArgumentError("ortho must be \"left\" or \"right\", got \"$ortho\""))
    sites = [siteind(rho, k) for k in order]
    outer = ortho == "left" ? first(ks) - 1 : last(ks)
    bond = 1 ≤ outer < length(rho) ? (linkind(rho, outer),) : ()
    worst = nothing
    for (k, s) in zip(order[1:(end - 1)], sites)
        tags = "Link,l=$(ortho == "left" ? k : k - 1)"
        F = ITensors.svd(phi, bond..., s; lefttags = tags, svd_kwargs...)
        rho[k], phi = F.U, F.S * F.V
        bond = (commonind(F.U, F.S),)
        worst = isnothing(worst) || F.spec.truncerr > worst.truncerr ? F.spec : worst
    end
    rho[last(order)] = phi
    return worst
end

"""
    steadystate(𝓛::MPO; nsweeps, maxdim, mindim, cutoff, kwargs...)
    steadystate(𝓛::MPO, rho0::MPS; ...)
    steadystate(𝓛::MPO, rho0::MPS, sweeps::Sweeps; ...)

DMRG for the steady state of the Lindbladian `𝓛`, an MPO on the vectorized sites of
`vectorized_siteinds`. Returns `(λ, rho)`: `rho` is the right null vector of `𝓛` with unit
trace, and `λ` its Rayleigh quotient, which vanishes at convergence, both up to
truncation.

The left state stays fixed at the vectorized identity, which `LindbladProjMPO` builds
from the site indices of `𝓛`, and only `rho` is swept, in the mixed canonical gauge.
Each window covers two sites, and every cut lies between physical sites, where the
identity factorizes and the test map of `LindbladProjMPO` exists whenever `tr ρ ≠ 0`. At
each window `sssolver` removes the traceless part of the window tensor, which is then
split back into site tensors by SVDs truncated with `maxdim`, `mindim` and `cutoff`.
`rho0` is any state on the sites of `𝓛` with nonzero trace. Without it, the sweeps start
from the maximally mixed state. The last form takes a `Sweeps` object, while the others build
it from the keywords.

`observer` is an ITensorMPS observer or a `BiorthoObserver`, which records the
vectorized identity as the left state. `outputlevel` is as in `ITensorMPS.dmrg`, and the
`linsolve_*` keywords are passed to `sssolver`. `noise` must be zero and indices must be
dense.
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
    n = _windowsize(P)
    solver_kwargs = (;
        tol = linsolve_tol, krylovdim = linsolve_krylovdim, maxiter = linsolve_maxiter,
        verbosity = linsolve_verbosity,
    )
    λ = zero(ComplexF64)
    for sw in 1:nsw
        stats = SweepStats()
        elapsed = @elapsed for (ℓ, ha) in _windowsweep(N, n)
            window = ℓ:(ℓ + n - 1)
            ITensorMPS.position!(P, rho, ℓ)
            λ, phi, solve_info = sssolver(P, prod(k -> rho[k], window); solver_kwargs...)
            spec = _replacewindow!(
                rho, window, phi;
                ortho = ha == 1 ? "left" : "right", maxdim = ITensorMPS.maxdim(sweeps, sw),
                mindim = ITensorMPS.mindim(sweeps, sw), cutoff = ITensorMPS.cutoff(sweeps, sw),
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