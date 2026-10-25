function BiorthoDMRG.MPIRealSpace(
        term::AbstractBiorthoProjMPO, comm::MPI.Comm;
        parts = nothing, pinv_rtol::Real = sqrt(eps(Float64)),
    )
    n, N = MPI.Comm_size(comm), length(term)
    parts = isnothing(parts) ? _equalparts(N, n) : collect(UnitRange{Int}, parts)
    length(parts) == n || throw(DimensionMismatch("$(length(parts)) parts for $n ranks"))
    tiles = first(first(parts)) == 1 && last(last(parts)) == N &&
        all(p -> first(parts[p + 1]) == last(parts[p]) + 1, 1:(n - 1))
    tiles || throw(ArgumentError("the parts must cover the sites 1:$N in order"))
    all(r -> length(r) ≥ 2, parts) || throw(ArgumentError("every part needs at least two sites"))
    return MPIRealSpace(term, comm, parts, Float64(pinv_rtol))
end

# Rank 0 brings the initial states into biorthonormal form centered on site 1 and
# broadcasts them, so that every rank holds the same right-biorthonormal tensors and link
# indices. Each rank moves the center to the first site of its part, which is exact for
# `basis = :shared` and, for `:biorthonormal`, under the condition of `biorthogonalize!`.
# Rank p takes the bond matrices at its right boundary from the center of rank p+1, and
# the odd ranks then move the center to the last site of their parts.
function _initialize(R::MPIRealSpace, chi0::MPS, psi0::MPS; kwargs...)
    rank, n = MPI.Comm_rank(R.comm), length(R.parts)
    a, e = first(R.parts[rank + 1]), last(R.parts[rank + 1])
    if rank == ROOT
        chi, psi = biorthogonalize!(copy(chi0), copy(psi0), 1; kwargs...)
    end
    chi, psi = MPI.bcast(rank == ROOT ? (chi, psi) : nothing, R.comm; root = ROOT)
    _moveright!(chi, psi, 1, a; kwargs...)
    rank > 0 && MPI.send((chi[a], psi[a]), R.comm; dest = rank - 1)
    Cinv = nothing
    if rank < n - 1
        chi_r, psi_r = MPI.recv(R.comm; source = rank + 1)
        Cinv = _bondinverses(chi_r * dag(psi[e + 1]), psi_r * dag(chi[e + 1]), R.pinv_rtol)
    end
    isodd(rank) && _moveright!(chi, psi, a, e; kwargs...)
    return chi, psi, Cinv
end

# Rank p updates the bond between site e, the last of its part, and site e+1, the first
# of rank p+1's. The initial guess joins the two centers with the pseudo-inverses of the
# bond matrices; the environments right of e+1 come from rank p+1, which receives its new
# center and the environments left of e+1.
function _perform!(R::MPIRealSpace, chi::MPS, psi::MPS, Cinv, e::Int; solver, solver_kwargs, split_kwargs)
    rank = MPI.Comm_rank(R.comm)
    chi_r, psi_r, renvs = MPI.recv(R.comm; source = rank + 1)
    _setright!(R.term, chi, psi, e + 2, renvs)
    ITensorMPS.position!(R.term, chi, psi, e)
    energy, phi_psi, phi_chi, solve_info =
        solver(R.term, psi[e] * Cinv.psi * psi_r, chi[e] * Cinv.chi * chi_r; solver_kwargs...)
    Cchi, Cpsi, bond_info = biorthosplitbond!(chi, psi, e, phi_chi, phi_psi; split_kwargs...)
    lenvs = _extendleft(R.term, chi, psi, e)
    MPI.send((Cchi * chi[e + 1], Cpsi * psi[e + 1], lenvs), R.comm; dest = rank + 1)
    chi[e], psi[e] = chi[e] * Cchi, psi[e] * Cpsi
    return energy, _bondinverses(Cchi, Cpsi, R.pinv_rtol), solve_info, bond_info
end

# Rank p+1's side of the update of the bond between site a-1 and site a, the first of its part.
function _serve!(R::MPIRealSpace, chi::MPS, psi::MPS, a::Int)
    rank = MPI.Comm_rank(R.comm)
    MPI.send((chi[a], psi[a], _extendright(R.term, chi, psi, a + 1)), R.comm; dest = rank - 1)
    chi[a], psi[a], lenvs = MPI.recv(R.comm; source = rank - 1)
    _setleft!(R.term, chi, psi, a - 1, lenvs)
    return chi, psi
end

function BiorthoDMRG.biorthodmrg(R::MPIRealSpace, chi0::MPS, psi0::MPS, sweeps::Sweeps; kwargs...)
    return _abortonerror(R.comm) do
        _realspacedmrg(R, chi0, psi0, sweeps; kwargs...)
    end
end

function _realspacedmrg(
        R::MPIRealSpace, chi0::MPS, psi0::MPS, sweeps::Sweeps;
        which::Symbol = :SR, target::Symbol = :which, howmany::Int = target === :overlap ? 2 : 1,
        basis::Symbol = :shared, biortho_cutoff::Real = sqrt(eps(Float64)), outputlevel::Int = 1,
        eigsolve_tol::Real = 1.0e-14, eigsolve_krylovdim::Int = 3, eigsolve_maxiter::Int = 1,
        eigsolve_verbosity::Int = 0, solver = biorthoproblemsolver,
    )
    nsw = ITensorMPS.nsweep(sweeps)
    any(sw -> ITensorMPS.noise(sweeps, sw) > 0, 1:nsw) &&
        throw(ArgumentError("noise is not defined for biorthogonal DMRG"))
    rank, n, PH = MPI.Comm_rank(R.comm), length(R.parts), R.term
    a, e = first(R.parts[rank + 1]), last(R.parts[rank + 1])
    chi, psi, Cinv = _initialize(R, chi0, psi0; basis, biortho_cutoff)
    ITensorMPS.position!(PH, chi, psi, isodd(rank) ? e - 1 : a)
    solver_kwargs = (;
        which, target, howmany, tol = eigsolve_tol, krylovdim = eigsolve_krylovdim,
        maxiter = eigsolve_maxiter, verbosity = eigsolve_verbosity,
    )
    energy = zero(ComplexF64)
    for sw in 1:nsw
        stats = SweepStats()
        split_kwargs = (;
            basis, biortho_cutoff, maxdim = ITensorMPS.maxdim(sweeps, sw),
            mindim = ITensorMPS.mindim(sweeps, sw), cutoff = ITensorMPS.cutoff(sweeps, sw),
        )
        elapsed = @elapsed for half in 1:2
            right = iseven(rank) == (half == 1)
            for ℓ in (right ? (a:(e - 1)) : ((e - 1):-1:a))
                ITensorMPS.position!(PH, chi, psi, ℓ)
                energy, phi_psi, phi_chi, solve_info =
                    solver(PH, psi[ℓ] * psi[ℓ + 1], chi[ℓ] * chi[ℓ + 1]; solver_kwargs...)
                bond_info = biorthoreplacebond!(
                    chi, psi, ℓ, phi_chi, phi_psi; ortho = right ? "left" : "right", split_kwargs...
                )
                stats = _merge_stats(stats, solve_info, bond_info)
            end
            if right && rank < n - 1
                energy, Cinv, solve_info, bond_info =
                    _perform!(R, chi, psi, Cinv, e; solver, solver_kwargs, split_kwargs)
                stats = _merge_stats(stats, solve_info, bond_info)
            elseif !right && rank > 0
                _serve!(R, chi, psi, a)
            end
        end
        reports = _gatherobjects((energy, stats, _maxlinkdim(psi, a:e)), R.comm)
        rank == ROOT && outputlevel ≥ 1 && _print_realspace_sweep(sw, reports, elapsed)
    end
    pieces = _gatherobjects((ITensor[chi[k] for k in a:e], ITensor[psi[k] for k in a:e], Cinv), R.comm)
    rank == ROOT || return nothing
    chi, psi = biorthogonalize!(_glue(pieces)..., 1; basis, biortho_cutoff)
    return energy, chi, psi
end