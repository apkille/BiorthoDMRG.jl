"""
    SweepStats

Worst-case diagnostics over the bonds of one sweep.

`maxtruncerr` is the largest truncation error of either state, `sigma_min` the
smallest cosine between the kept left and right bases, `cosangle_min` the
smallest cosine between the local left and right eigenvectors before
normalization, and `maxresidual` the largest relative residual of a local left or
right eigenvector. `SweepStats()` is the starting value before any bond is
visited.
"""
struct SweepStats
    maxtruncerr::Float64
    sigma_min::Float64
    cosangle_min::Float64
    maxresidual::Float64
end

SweepStats() = SweepStats(0.0, Inf, 1.0, 0.0)

function _merge_stats(s::SweepStats, solve_info, bond_info)
    return SweepStats(
        max(s.maxtruncerr, bond_info.spec_psi.truncerr, bond_info.spec_chi.truncerr),
        min(s.sigma_min, bond_info.sigma_min),
        min(s.cosangle_min, solve_info.cosangle),
        max(s.maxresidual, solve_info.residual),
    )
end

"""
    BiorthoObserver(; H=nothing, residual_tol=0.0)

Observer for `biorthodmrg` that records one entry per sweep. `energies` holds the
energy, `conditions` the condition number `‖chi‖‖psi‖/|⟨chi|psi⟩|`, and `stats`
the `SweepStats`.

If `H` is given, `residuals` holds the relative residuals of `psi` and `chi`,
computed with `H` and its adjoint `Hdag`, and the run stops after the first sweep
in which both are below `residual_tol`. Without `H`, the run never stops early.

Residuals are used instead of energy changes because the energy need not converge
monotonically. They resolve values only down to about √eps, so `residual_tol`
must be larger than that.
"""
struct BiorthoObserver
    H::Union{MPO, Nothing}
    Hdag::Union{MPO, Nothing}
    residual_tol::Float64
    energies::Vector{ComplexF64}
    residuals::Vector{NTuple{2, Float64}}
    conditions::Vector{Float64}
    stats::Vector{SweepStats}
end

function BiorthoObserver(; H::Union{MPO, Nothing} = nothing, residual_tol::Real = 0.0)
    Hdag = isnothing(H) ? nothing : adjoint_mpo(H)
    return BiorthoObserver(H, Hdag, residual_tol, ComplexF64[], NTuple{2, Float64}[], Float64[], SweepStats[])
end

_record_sweep!(observer, energy, chi, psi, stats; kwargs...) = nothing

function _record_sweep!(o::BiorthoObserver, energy, chi, psi, stats; outputlevel::Int = 0)
    push!(o.energies, energy)
    push!(o.stats, stats)
    push!(o.conditions, condition_number(chi, psi))
    if !isnothing(o.H)
        r = residuals(o.H, chi, psi, energy; Hdag = o.Hdag)
        push!(o.residuals, r)
        outputlevel ≥ 1 &&
            @printf("  residuals: psi=%.2E chi=%.2E  kappa=%.3E\n", r[1], r[2], last(o.conditions))
    end
    return o
end

ITensorMPS.measure!(::BiorthoObserver; kwargs...) = nothing

function ITensorMPS.checkdone!(o::BiorthoObserver; kwargs...)
    return !isempty(o.residuals) && maximum(last(o.residuals)) < o.residual_tol
end