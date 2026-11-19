"""
    BiorthoDMRG

Two-site biorthogonal DMRG for non-Hermitian MPOs.  With |ψ⟩ = P_{ℓψ}|φ_ψ⟩ and
|χ⟩ = P_{ℓχ}|φ_χ⟩ in the biorthonormal gauge P_{ℓχ}†P_{ℓψ} = 1, each bond update of
`biorthodmrg` solves

    P_{ℓχ}† H P_{ℓψ} |φ_ψ⟩ = E |φ_ψ⟩,     P_{ℓψ}† H† P_{ℓχ} |φ_χ⟩ = Ē |φ_χ⟩,

normalizes ⟨φ_χ|φ_ψ⟩ = 1, and refactorizes both blocks, in one shared orthonormal basis
or in a biorthonormal pair of bases.

For the steady state of a Lindbladian 𝓛, `steadystate` holds the left state fixed at the
vectorized identity |1⟩, which the vectorized site indices determine, and keeps
|ρ⟩ = P_{ℓρ}|φ_ρ⟩ in the mixed canonical gauge. With the test map P_{ℓ1} of
`LindbladProjMPO`, which satisfies P_{ℓ1}†P_{ℓρ} = 1 and P_{ℓ1}|φ_1⟩ = |1⟩ for
|φ_1⟩ = P_{ℓρ}†|1⟩, each bond update solves

    P_{ℓ1}† 𝓛 P_{ℓρ} |δφ⟩ = P_{ℓ1}† 𝓛 P_{ℓρ} |φ_ρ⟩,     ⟨φ_1|δφ⟩ = 0,

and replaces |φ_ρ⟩ by |φ_ρ⟩ - |δφ⟩.
"""
module BiorthoDMRG

using ITensors: ITensors, ITensor, Index, addtags, array, commonind, dag, delta, dense, dim,
    hascommoninds, hasqns, hassameinds, itensor, noprime, onehot, prime, replaceind,
    replaceinds, scalar, sim, swapprime, uniqueinds
using ITensorMPS: ITensorMPS, MPO, MPS, Sweeps, firstsiteinds, inner, linkind, maxlinkdim,
    orthogonalize, siteind
using KrylovKit: EigSorter, eigsolve, linsolve
using LinearAlgebra: LinearAlgebra, Diagonal, norm, rmul!
using Printf: @printf

export AbstractBiorthoProjMPO, BiorthoObserver, BiorthoProjMPO, BiorthoProjMPO_MPS,
    BiorthoProjMPS, BiorthoProjSum, LindbladProjMPO, MPIBiorthoSum, MPIObserver, adjoint_mpo,
    adjoint_product, biorthodmrg, biorthogonalize!, biorthoproblemsolver, sssolver,
    biorthoreplacebond!, condition_number, residuals, serve!, steadystate, stopservice!,
    vectorize, vectorized_identity, vectorized_siteinds

include("environments.jl")
include("doubled.jl")
include("projmpo.jl")
include("projmps.jl")
include("projsum.jl")
include("projlindblad.jl")
include("factorization.jl")
include("gauge.jl")
include("solver.jl")
include("diagnostics.jl")
include("observer.jl")
include("parallel.jl")
include("bodmrg.jl")
include("steadystate.jl")
include("models.jl")

end