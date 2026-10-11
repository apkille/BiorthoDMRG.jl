function _print_sweep(sw::Int, energy, psi::MPS, s::SweepStats, elapsed::Real)
    @printf(
        "After sweep %d energy=%s maxlinkdim=%d maxerr=%.2E sigma_min=%.2E cosangle_min=%.2E time=%.3f\n",
        sw, energy, maxlinkdim(psi), s.maxtruncerr, s.sigma_min, s.cosangle_min, elapsed
    )
    return nothing
end

"""
    biorthodmrg(H::MPO, psi0::MPS; nsweeps, maxdim, mindim, cutoff, kwargs...)
    biorthodmrg(H::MPO, chi0::MPS, psi0::MPS; ...)
    biorthodmrg(H::MPO, refs::Vector{Tuple{MPS,MPS}}, chi0::MPS, psi0::MPS; weight, ...)
    biorthodmrg(PH::AbstractBiorthoProjMPO, chi0::MPS, psi0::MPS, sweeps::Sweeps; ...)

Two-site DMRG for the left and right eigenvectors of a non-Hermitian `H`. Returns
`(E, chi, psi)`: `psi` is the right eigenvector and `chi` the left eigenvector,
normalized so that `⟨chi|psi⟩ = 1`, and `E = ⟨chi|H|psi⟩`, all up to truncation.

With one initial state, `chi0 = psi0`. With `refs`, previously converged pairs
`(chi_i, psi_i)` are deflated by shifting their eigenvalues by `weight`. The last
form takes any projected operator and a `Sweeps` object; the others build both.

`which` selects the eigenvalue as in `KrylovKit.eigsolve`. `target` and `howmany`
choose among Ritz pairs as in `biorthoproblemsolver`. `basis` chooses `:shared`
(stable, larger bond dimension) or `:biorthonormal` (compact, unbounded local
spectrum); see `biorthoreplacebond!`. `biortho_cutoff` drops cosines between the
kept left and right bases at or below it, and applies only to `:biorthonormal`.
`observer` is an ITensorMPS observer or a `BiorthoObserver`. `outputlevel` and
the `eigsolve_*` keywords are as in `ITensorMPS.dmrg`. `noise` must be zero and
indices must be dense.
"""
function biorthodmrg(
        PH::AbstractBiorthoProjMPO, chi0::MPS, psi0::MPS, sweeps::Sweeps;
        which::Symbol = :SR, target::Symbol = :which, howmany::Int = target === :overlap ? 2 : 1,
        basis::Symbol = :shared, biortho_cutoff::Real = sqrt(eps(Float64)),
        observer = ITensorMPS.NoObserver(), outputlevel::Int = 1, eigsolve_tol::Real = 1.0e-14,
        eigsolve_krylovdim::Int = 3, eigsolve_maxiter::Int = 1, eigsolve_verbosity::Int = 0,
    )
    nsw = ITensorMPS.nsweep(sweeps)
    any(sw -> ITensorMPS.noise(sweeps, sw) > 0, 1:nsw) &&
        throw(ArgumentError("noise is not defined for biorthogonal DMRG"))
    N = length(psi0)
    N ≥ 2 || throw(ArgumentError("biorthodmrg requires at least two sites"))
    length(PH) == N ||
        throw(DimensionMismatch("the operator and the initial states have different lengths"))
    chi, psi = biorthogonalize!(copy(chi0), copy(psi0), 1)
    ITensorMPS.position!(PH, chi, psi, 1)
    solver_kwargs = (;
        which, target, howmany, tol = eigsolve_tol, krylovdim = eigsolve_krylovdim,
        maxiter = eigsolve_maxiter, verbosity = eigsolve_verbosity,
    )
    energy = zero(ComplexF64)
    for sw in 1:nsw
        stats = SweepStats()
        elapsed = @elapsed for (ℓ, ha) in ITensorMPS.sweepnext(N)
            ITensorMPS.position!(PH, chi, psi, ℓ)
            energy, phi_psi, phi_chi, solve_info =
                biorthoproblemsolver(PH, psi[ℓ] * psi[ℓ + 1], chi[ℓ] * chi[ℓ + 1]; solver_kwargs...)
            bond_info = biorthoreplacebond!(
                chi, psi, ℓ, phi_chi, phi_psi;
                ortho = ha == 1 ? "left" : "right", basis, biortho_cutoff,
                maxdim = ITensorMPS.maxdim(sweeps, sw), mindim = ITensorMPS.mindim(sweeps, sw),
                cutoff = ITensorMPS.cutoff(sweeps, sw),
            )
            stats = _merge_stats(stats, solve_info, bond_info)
            ITensorMPS.measure!(
                observer; energy, psi, projected_operator = PH, bond = ℓ, sweep = sw,
                half_sweep = ha, spec = bond_info.spec_psi, outputlevel, sweep_is_done = (ℓ == 1 && ha == 2),
            )
        end
        outputlevel ≥ 1 && _print_sweep(sw, energy, psi, stats, elapsed)
        _record_sweep!(observer, energy, chi, psi, stats; outputlevel)
        ITensorMPS.checkdone!(observer; energy, psi, sweep = sw, outputlevel) && break
    end
    return energy, chi, psi
end

function biorthodmrg(H::MPO, chi0::MPS, psi0::MPS, sweeps::Sweeps; kwargs...)
    return biorthodmrg(BiorthoProjMPO(H), chi0, psi0, sweeps; kwargs...)
end

biorthodmrg(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...) = biorthodmrg(H, psi0, psi0, sweeps; kwargs...)

function biorthodmrg(
        H::MPO, refs::Vector{<:Tuple{MPS, MPS}}, chi0::MPS, psi0::MPS, sweeps::Sweeps;
        weight::Number, kwargs...,
    )
    return biorthodmrg(BiorthoProjMPO_MPS(H, refs; weight), chi0, psi0, sweeps; kwargs...)
end

function biorthodmrg(
        args...; nsweeps::Int, maxdim = ITensorMPS.default_maxdim(),
        mindim = ITensorMPS.default_mindim(), cutoff = ITensorMPS.default_cutoff(Float64),
        noise = ITensorMPS.default_noise(), kwargs...,
    )
    sweeps = ITensorMPS._dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
    return biorthodmrg(args..., sweeps; kwargs...)
end