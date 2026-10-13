"""
    Atoms

Two-level atom arrays with free-space dipole-dipole couplings.

Here we consider the steady state of the vectorized Liouvillian ([`steadystate`](@ref)) and eigenstates of
the effective non-Hermitian Hamiltonian ([`eigenstate`](@ref)).
"""
module Atoms

import HDF5, ITensors, ITensorMPS, LinearAlgebra
using ..BiorthoDMRG: BiorthoObserver, biorthodmrg, condition_number, sssolver
using ..Models: vectorized_identity

export K0, LAMBDA, greens_coupling, interactions,
    Biortho, Solution, sem,
    Model, chain, pumped, driven, shifted, quadratic, linear, alternating,
    supergaussian, natoms, positions, couplings,
    Population, Emission, Coherence, Density, Correlation, Pairing, Density2, Inversion2,
    coherences, densities, correlations, pairings, densities2, inversions2,
    allground, allexcited, single_excitation,
    steadystate, eigenstate,
    cvector, cmatrix, connected, g2r, mmatrix, mconnected, mode, g2matrix, zzmatrix, fano,
    save, load

include("greens.jl")
include("types.jl")
include("models.jl")
include("operators.jl")
include("ansatzes.jl")
include("simulate.jl")
include("analysis.jl")
include("io.jl")

end