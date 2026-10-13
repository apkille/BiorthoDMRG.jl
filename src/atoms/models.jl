"""
    AbstractModel

A physical system: atomic geometry plus any external drives. Representation-free —
[`couplings`](@ref) reduces any model to the matrices the backends consume.
"""
abstract type AbstractModel end

"""
    Model(; pos, dipole = [0,0,1])

Bare dipole-dipole system: `N` two-level atoms at positions `pos` (3-vectors) sharing
the unit dipole orientation `dipole`. No pump, no drive.

Lengths are in units of `c/Γ₀`; one wavelength is [`LAMBDA`](@ref) ≈ 2.9746.
"""
Base.@kwdef struct Model{P,D} <: AbstractModel
    pos::P
    dipole::D = [0.0, 0.0, 1.0]
end

"""
    Pumped(base, rate)

`base` with a local incoherent pump at rate `w = rate` on every atom, adding
`w (σ⁺ᵢ ρ σ⁻ᵢ - ½{σ⁻ᵢσ⁺ᵢ, ρ})` to the Liouvillian. Build with [`pumped`](@ref).
"""
struct Pumped{M<:AbstractModel} <: AbstractModel
    base::M
    rate::Float64
end

"""
    Driven(base, rabi, detuning, kvec)

`base` under a coherent plane-wave drive, in the frame rotating at the laser
frequency. Build with [`driven`](@ref).
"""
struct Driven{M<:AbstractModel,K} <: AbstractModel
    base::M
    rabi::Float64
    detuning::Float64
    kvec::K
end

"""
    pumped(m; rate) -> Pumped

Add a local incoherent pump at `rate` to `m`. Composes with [`driven`](@ref) in
either order.
"""
pumped(m::AbstractModel; rate::Real) = Pumped(m, float(rate))

"""
    driven(m; rabi, detuning = 0.0, kvec = zeros(3)) -> Driven

Add a coherent plane-wave drive to `m`:

```math
H_\\mathrm{drive} = -\\Delta \\sum_i \\sigma^+_i \\sigma^-_i
    + \\sum_i \\left( \\Omega_i \\sigma^+_i + \\Omega_i^* \\sigma^-_i \\right),
\\qquad \\Omega_i = \\tfrac{\\Omega}{2} e^{i \\mathbf{k}_L \\cdot \\mathbf{r}_i}
```

`rabi` is the Rabi frequency `Ω` (the per-atom coupling is `Ω/2`). `detuning` is
`Δ = ω_L - ω₀`, so **positive `detuning` is a blue-detuned drive**; it enters as `-Δ`
on the diagonal of `coh`. `kvec` is the laser wavevector: `zeros(3)` drives all atoms
in phase, while `K0 .* n̂` imprints the phase gradient of a beam along `n̂`.
"""
driven(m::AbstractModel; rabi::Real, detuning::Real = 0.0, kvec = zeros(3)) =
    Driven(m, float(rabi), float(detuning), kvec)

"""
    chain(N, d; axis = [1,0,0]) -> Vector{Vector{Float64}}

`N` equally spaced positions with separation `d` along `axis`, starting at the origin.
`axis` is normalized, so `d` is the true spacing. `d` is in units of `c/Γ₀` — use
`d = x * LAMBDA` for a spacing of `x` wavelengths.
"""
function chain(N::Int, d::Real; axis = [1.0, 0.0, 0.0])
    â = axis ./ LinearAlgebra.norm(axis)
    return [(i-1)*d .* â for i in 1:N]
end

"""
    Shifted(base, shifts)

`base` with per-atom energy shifts, adding `∑ᵢ Vᵢ σ⁺ᵢσ⁻ᵢ` to `H`. Build with
[`shifted`](@ref) or [`quadratic`](@ref).
"""
struct Shifted{M<:AbstractModel} <: AbstractModel
    base::M
    shifts::Vector{Float64}
end

"""
    shifted(m; shifts) -> Shifted

Add arbitrary per-atom energy shifts `Vᵢ`. Composes with [`pumped`](@ref),
[`driven`](@ref), and further `shifted` calls in any order (shifts accumulate).
"""
function shifted(m::AbstractModel; shifts::AbstractVector{<:Real})
    length(shifts) == natoms(m) ||
        error("shifts has $(length(shifts)) entries for $(natoms(m)) atoms")
    return Shifted(m, collect(float.(shifts)))
end

# axial coordinate of each atom relative to `center`, along the normalized `axis`
_axial(m::AbstractModel, axis, center) =
    (â = axis ./ LinearAlgebra.norm(axis);
     [LinearAlgebra.dot(â, r .- center) for r in positions(m)])

"""
     quadratic(m; kappa, axis = [0,0,1], center = zeros(3)) -> Shifted
 
Quadratic light-shift potential `Vᵢ = κ (â ⋅ (rᵢ - center))²` along the normalized
`axis`. `kappa` is in units of `Γ₀³/c²`.
"""
quadratic(m::AbstractModel; kappa::Real, axis = [0.0, 0.0, 1.0], center = zeros(3)) =
    shifted(m; shifts = kappa .* _axial(m, axis, center) .^ 2)
 
"""
    linear(m; gradient, axis = [0,0,1], center = zeros(3)) -> Shifted

Linear light-shift potential `Vᵢ = F (â ⋅ (rᵢ - center))`, i.e. a uniform force. `center`
sets only the offset, since a constant shift is a global energy origin. `gradient` is in
units of `Γ₀²/c`.
"""
linear(m::AbstractModel; gradient::Real, axis = [0.0, 0.0, 1.0], center = zeros(3)) =
    shifted(m; shifts = gradient .* _axial(m, axis, center))

"""
    alternating(m; amplitude, period, phase = 0.0, axis = [0,0,1], center = zeros(3))
        -> Shifted

Sinusoidal light-shift potential `Vᵢ = V₀ cos(2π zᵢ/period + phase)` with
`zᵢ = â ⋅ (rᵢ - center)`. `period` is a length; for a standing wave of wavelength `λ`
take `period = λ/2`. `amplitude` is in units of `Γ₀`.

A staggered potential is the special case `period = 2d` with `phase` chosen so the atoms
sample the extrema, giving `Vᵢ = ±V₀`. That imparts momentum `π/d` and so couples the
zone center to the band edge.
"""
alternating(m::AbstractModel; amplitude::Real, period::Real, phase::Real = 0.0,
            axis = [0.0, 0.0, 1.0], center = zeros(3)) =
    shifted(m; shifts = amplitude .* cos.(2π .* _axial(m, axis, center) ./ period .+ phase))

"""
    supergaussian(m; depth, waist, order = 1, axis = [0,0,1], center = zeros(3)) -> Shifted

Super-Gaussian light shift

    Vᵢ = U [1 - exp(-2 (zᵢ/w₀)^(2p))],    zᵢ = â ⋅ (rᵢ - center),

with `depth` = `U` in units of `Γ₀`, `waist` = `w₀`, and `order` = `p`. `V = 0` at the
trap center and `V → U` outside.

`order = 1` is the Gaussian light shift of a red-detuned beam; large `order` approaches a
flat-bottomed box of half-width `w₀`.
"""
supergaussian(m::AbstractModel; depth::Real, waist::Real, order::Int = 1,
            axis = [0.0, 0.0, 1.0], center = zeros(3)) =
    shifted(m; shifts = depth .*
        (1 .- exp.(-2 .* (_axial(m, axis, center) ./ waist) .^ (2order))))

"""
    natoms(m) -> Int
    positions(m) -> Vector

Number of atoms and their positions, forwarded through any drive decorators.
"""
natoms(m::Model) = length(m.pos)
natoms(m::Union{Pumped,Driven,Shifted}) = natoms(m.base)
positions(m::Model) = m.pos
positions(m::Union{Pumped,Driven,Shifted}) = positions(m.base)

"""
    Couplings(coh, decay, pump, drive)

Everything the backends need, in representation-free form: coherent exchange
`J_ij = coh`, dissipative `Γ_ij = decay`, incoherent pump rate `pump`, and per-atom
coherent drive amplitudes `drive` (`Ωᵢ`, empty when undriven). Detuning is folded
into the diagonal of `coh`.
"""
Base.@kwdef struct Couplings
    coh::Matrix{Float64}
    decay::Matrix{Float64}
    pump::Float64 = 0.0
    drive::Vector{ComplexF64} = ComplexF64[]
end

"""
    couplings(m::AbstractModel) -> Couplings

Reduce a model to its coupling matrices. Decorators accumulate onto the base result:
`Pumped` adds to `pump`, `Driven` shifts `coh`'s diagonal and adds to `drive`, so
nesting order does not matter.
"""
function couplings(m::Model)
    coh, decay = interactions(m.pos, m.dipole)
    return Couplings(coh, decay, 0.0, ComplexF64[])
end
function couplings(m::Pumped)
    c = couplings(m.base)
    return Couplings(c.coh, c.decay, c.pump + m.rate, c.drive)
end
function couplings(m::Driven)
    c = couplings(m.base)
    coh = copy(c.coh)
    for i in axes(coh, 1)
        coh[i,i] -= m.detuning
    end
    r = positions(m)
    Ω = [0.5 * m.rabi * cis(LinearAlgebra.dot(m.kvec, r[i])) for i in eachindex(r)]
    return Couplings(coh, c.decay, c.pump, isempty(c.drive) ? Ω : c.drive .+ Ω)
end
function couplings(m::Shifted)
    c = couplings(m.base)
    coh = copy(c.coh)
    for i in axes(coh, 1)
        coh[i,i] += m.shifts[i]
    end
    return Couplings(coh, c.decay, c.pump, c.drive)
end