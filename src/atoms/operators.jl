register(::Type{KetRegister}, N::Int) = KetRegister(ITensorMPS.siteinds("S=1/2", N))
function register(::Type{DoubledRegister}, N::Int)
    s = ITensorMPS.siteinds("S=1/2", 2N)
    return DoubledRegister(s, vectorized_identity(s))
end

function hamiltonian(r::KetRegister, c::Couplings; tol::Real = 1e-12)
    N = natoms(r)
    @assert size(c.coh) == (N, N)
    os = ITensorMPS.OpSum()
    for i in 1:N, j in 1:N
        abs(c.coh[i,j]) > tol || continue
        i == j ? (os += (c.coh[i,i], "ProjUp", i)) :
                 (os += (c.coh[i,j], "S+", i, "S-", j))
    end
    for i in eachindex(c.drive)
        abs(c.drive[i]) > tol || continue
        os += (c.drive[i], "S+", i)
        os += (conj(c.drive[i]), "S-", i)
    end
    return ITensorMPS.MPO(os, r.sites)
end

# os + a f(H_nh) on the sites site(1), …, site(N); f = identity, or conj for the
# complex-conjugated coefficients
function _nh_terms(os, c::Couplings, site, a::Number, f)
    N = size(c.coh, 1)
    for i in 1:N, j in 1:N
        g = c.coh[i,j] - 0.5im * c.decay[i,j]
        abs(g) > 0.0 || continue
        if i == j
            os += (a * f(g), "ProjUp", site(i))
        else
            os += (a * f(g), "S+", site(i), "S-", site(j))
        end
    end
    for i in eachindex(c.drive)
        abs(c.drive[i]) > 0.0 || continue
        os += (a * f(c.drive[i]), "S+", site(i))
        os += (a * f(conj(c.drive[i])), "S-", site(i))
    end
    if c.pump != 0.0
        for i in 1:N
            os += (a * f(-0.5im * c.pump), "ProjDn", site(i))
        end
    end
    return os
end

# effective non-Hermitian Hamiltonian
function hamiltonian_nh(r::KetRegister, c::Couplings)
    N = natoms(r)
    @assert size(c.coh) == size(c.decay) == (N, N)
    return ITensorMPS.MPO(_nh_terms(ITensorMPS.OpSum(), c, identity, 1.0, identity), r.sites)
end

# jump channels
function _decay_modes(decay::AbstractMatrix; tol::Real = 1e-12)
    Λ, U = LinearAlgebra.eigen(LinearAlgebra.Hermitian(Matrix(decay)))
    minimum(Λ) < -tol && @warn "decay matrix is not positive semidefinite" λmin=minimum(Λ)
    idx = findall(>(tol), Λ)
    return Λ[idx], U[:, idx]
end

"""
    liouvillian(sites, c; cutoff = 1e-14, tol = 1e-12)

The Liouvillian of `c` on the doubled lattice `sites`, with the ket of atom `i` on site
`2i-1` and its bra on site `2i`:

    ℒ = -i H_nh ⊗ 1 + i 1 ⊗ conj(H_nh) + Σᵢⱼ Γᵢⱼ σ⁻ᵢ ⊗ σ⁻ⱼ + w Σᵢ σ⁺ᵢ ⊗ σ⁺ᵢ,

where `conj(H_nh)` has complex-conjugated coefficients, `w = c.pump`, and `Γ` is
`c.decay` restricted to its modes above `tol`, so `ℒ` is completely positive and trace
preserving. The MPO is compressed with `cutoff`.
"""
function liouvillian(sites::Vector{<:ITensors.Index}, c::Couplings;
                     cutoff::Real = 1e-14, tol::Real = 1e-12)
    N = length(sites) ÷ 2
    @assert size(c.coh) == size(c.decay) == (N, N)
    Λ, U = _decay_modes(c.decay; tol)
    Γ = U * LinearAlgebra.Diagonal(Λ) * U'
    ct = Couplings(c.coh, (Γ + Γ') / 2, c.pump, c.drive)
    ket(i) = 2i - 1
    bra(i) = 2i
    os = _nh_terms(ITensorMPS.OpSum(), ct, ket, -1.0im, identity)
    os = _nh_terms(os, ct, bra, 1.0im, conj)
    for i in 1:N, j in 1:N
        abs(ct.decay[i,j]) > 0.0 || continue
        os += (ct.decay[i,j], "S-", ket(i), "S-", bra(j))
    end
    if c.pump != 0.0
        for i in 1:N
            os += (c.pump, "S+", ket(i), "S+", bra(i))
        end
    end
    return ITensorMPS.MPO(os, sites; cutoff)
end
liouvillian(r::DoubledRegister, c::Couplings; kw...) = liouvillian(r.sites, c; kw...)

# observables
function population(r::KetRegister)
    os = ITensorMPS.OpSum()
    for i in 1:natoms(r); os += (1.0, "ProjUp", i); end
    return ITensorMPS.MPO(os, r.sites)
end
function population(r::DoubledRegister)
    os = ITensorMPS.OpSum()
    for i in 1:natoms(r); os += (1.0, "ProjUp", 2i - 1); end
    return ITensorMPS.MPO(os, r.sites)
end

function emission(r::KetRegister, decay; tol::Real = 1e-12)
    os = ITensorMPS.OpSum()
    for i in 1:natoms(r), j in 1:natoms(r)
        abs(decay[i,j]) > tol || continue
        i == j ? (os += (decay[i,i], "ProjUp", i)) : (os += (decay[i,j], "S+", i, "S-", j))
    end
    return ITensorMPS.MPO(os, r.sites)
end
function emission(r::DoubledRegister, decay; tol::Real = 1e-12)
    os = ITensorMPS.OpSum()
    N = natoms(r)
    for i in 1:N, j in 1:N
        abs(decay[i,j]) > tol || continue
        i == j ? (os += (decay[i,i], "ProjUp", 2i-1)) :
                 (os += (decay[i,j], "S+", 2i-1, "S-", 2j-1))
    end
    return ITensorMPS.MPO(os, r.sites)
end

# states and expectations
_itname(s::Symbol) = s === :e ? "Up" : s === :g ? "Dn" : error("state must be :e or :g")
initial_state(r::Register, spec::Symbol) = initial_state(r, fill(spec, natoms(r)))

initial_state(r::KetRegister, spec::Vector{Symbol}) = ITensorMPS.MPS(r.sites, _itname.(spec))
initial_state(r::DoubledRegister, spec::Vector{Symbol}) =
    ITensorMPS.MPS(r.sites, repeat(_itname.(spec); inner = 2))

# site-resolved observables
const MPSRegister = Union{KetRegister,DoubledRegister}

# atom i lives on site i of a ket lattice, ket site 2i-1 of the doubled lattice
_site(::KetRegister, i::Int)     = i
_site(::DoubledRegister, i::Int) = 2i - 1

"""
    coherence(reg, i)

`σ⁻ᵢ`, giving the dipole amplitude `vᵢ = ⟨σ⁻ᵢ⟩`. Non-Hermitian: nonzero and
laser-locked under a coherent drive, identically zero in the ensemble under a
U(1)-symmetric Liouvillian.
"""
function coherence(r::MPSRegister, i::Int)
    os = ITensorMPS.OpSum()
    os += (1.0, "S-", _site(r, i))
    return ITensorMPS.MPO(os, r.sites)
end

"""
    density(reg, i)

Site population `nᵢ = σ⁺ᵢσ⁻ᵢ`.
"""
function density(r::MPSRegister, i::Int)
    os = ITensorMPS.OpSum()
    os += (1.0, "ProjUp", _site(r, i))
    return ITensorMPS.MPO(os, r.sites)
end

"""
    correlation(reg, i, j)

`σ⁺ᵢσ⁻ⱼ`, giving the one-body matrix `Cᵢⱼ`. `C = C† ⪰ 0`, so only `j ≥ i` is
needed; the diagonal reduces to [`density`](@ref).
"""
function correlation(r::MPSRegister, i::Int, j::Int)
    i == j && return density(r, i)
    os = ITensorMPS.OpSum()
    os += (1.0, "S+", _site(r, i), "S-", _site(r, j))
    return ITensorMPS.MPO(os, r.sites)
end

"""
    pairing(reg, i, j)

`σ⁻ᵢσ⁻ⱼ`, giving the anomalous matrix `Mᵢⱼ`. Symmetric, and `i == j` is rejected
because `σ⁻σ⁻ = 0` on a two-level atom.
"""
function pairing(r::MPSRegister, i::Int, j::Int)
    i == j && error("pairing: σ⁻ᵢσ⁻ᵢ = 0; M has no diagonal")
    os = ITensorMPS.OpSum()
    os += (1.0, "S-", _site(r, i), "S-", _site(r, j))
    return ITensorMPS.MPO(os, r.sites)
end

"""
    density2(reg, i, j)

`nᵢnⱼ`. For `i ≠ j` the site operators commute, so this is already normally
ordered and `g⁽²⁾ᵢⱼ = ⟨nᵢnⱼ⟩/(⟨nᵢ⟩⟨nⱼ⟩)`. For `i == j` it collapses to `⟨nᵢ⟩`
because `nᵢ² = nᵢ` — that is the second moment, **not** `g⁽²⁾(0)`, which is
exactly zero for a two-level atom.
"""
function density2(r::MPSRegister, i::Int, j::Int)
    os = ITensorMPS.OpSum()
    i == j ? (os += (1.0, "ProjUp", _site(r, i))) :
             (os += (1.0, "ProjUp", _site(r, i), "ProjUp", _site(r, j)))
    return ITensorMPS.MPO(os, r.sites)
end

"""
    inversion2(reg, i, j)

`σᶻᵢσᶻⱼ`. Redundant with [`density2`](@ref) in the mean, since
`σᶻᵢσᶻⱼ = 4nᵢnⱼ - 2nᵢ - 2nⱼ + 1`.
"""
function inversion2(r::MPSRegister, i::Int, j::Int)
    i == j && error("inversion2: σᶻᵢσᶻᵢ = 1")
    os = ITensorMPS.OpSum()
    os += (4.0, "Sz", _site(r, i), "Sz", _site(r, j))
    return ITensorMPS.MPO(os, r.sites)
end

expect(::KetRegister, O::ITensorMPS.MPO, psi::ITensorMPS.MPS) =
    ITensorMPS.inner(psi', O, psi) / ITensorMPS.inner(psi, psi)
expect(r::DoubledRegister, O::ITensorMPS.MPO, rho::ITensorMPS.MPS) =
    ITensorMPS.inner(r.id', O, rho) / ITensorMPS.inner(r.id, rho)