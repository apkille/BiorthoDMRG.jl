"""
    steadystate(backend::Biortho, model; observables = (Population(),))

Observables of the stationary state of `model`, from `biorthodmrg` with `solver = sssolver`
on the vectorized Liouvillian `ℒ`. The vectorized identity is an exact left null vector of
`ℒ`, and the steady state is the right null vector. Sweeps start from the
infinite-temperature state. Returns `(sol, res)` with `sol.t == [Inf]` and
`res = (; eigenvalue, residuals, kappa)`: `eigenvalue` vanishes at convergence, `residuals`
are those of `ρ` and of the identity, and `kappa = √(2ᴺ tr ρ²)/|tr ρ|`.
"""
function steadystate(
    backend::Biortho, model::AbstractModel;
    observables = (Population(),),
)
    obs = observables isa Observable ? (observables,) : observables
    reg = register(DoubledRegister, natoms(model))
    c   = couplings(model)
    ops = [materialize(o, reg, c) for o in obs]
    L   = liouvillian(reg, c; cutoff = backend.mpo_cutoff)
    observer = BiorthoObserver(; H = L, backend.residual_tol)
    λ, _, rho = biorthodmrg(L, reg.id; solver = sssolver, observer, _opts(backend)...)
    data = permutedims(ComplexF64[expect(reg, O, rho) for O in ops])
    res = (; eigenvalue = λ, residuals = collect(last(observer.residuals)),
             kappa = last(observer.conditions))
    return Solution([Inf], Symbol[name(o) for o in obs], data, nothing, backend), res
end

"""
    eigenstate(backend::Biortho, model; state, observables = (Population(),), which = :LI)

Observables of the right eigenvector `ψ` of the effective non-Hermitian Hamiltonian
`H_nh` of `model` selected by `which`, from `biorthodmrg`. `state` is the DMRG guess. At
zero Rabi frequency `H_nh` conserves the excitation number, so a guess in one sector, for
example [`single_excitation`](@ref), stays in it up to rounding. Observables are
`⟨ψ|O|ψ⟩/⟨ψ|ψ⟩`; for `which = :LI` this is the `t → ∞` limit of the normalized no-jump
evolution. Returns `(sol, res)` with `sol.t == [Inf]` and `res = (; energy, residuals,
kappa)`, where `kappa = ‖χ‖‖ψ‖/|⟨χ|ψ⟩|`.
"""
function eigenstate(
    backend::Biortho, model::AbstractModel;
    state,
    observables = (Population(),),
    which::Symbol = :LI,
)
    state isa Symbol || length(state) == natoms(model) ||
    error("state has $(length(state)) entries for $(natoms(model)) atoms")
    obs = observables isa Observable ? (observables,) : observables
    reg = register(KetRegister, natoms(model))
    c   = couplings(model)
    ops = [materialize(o, reg, c) for o in obs]
    Hnh = hamiltonian_nh(reg, c)
    observer = BiorthoObserver(; H = Hnh, backend.residual_tol)
    E, chi, psi = biorthodmrg(Hnh, initial_state(reg, state); which, observer, _opts(backend)...)
    data = permutedims(ComplexF64[expect(reg, O, psi) for O in ops])
    res = (; energy = E, residuals = collect(last(observer.residuals)),
             kappa = condition_number(chi, psi))
    return Solution([Inf], Symbol[name(o) for o in obs], data, nothing, backend), res
end