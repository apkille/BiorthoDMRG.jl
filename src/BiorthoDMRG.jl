"""
    BiorthoDMRG

Two-site biorthogonal DMRG for non-Hermitian MPOs.  With |ψ⟩ = P_{ℓψ}|φ_ψ⟩ and
|χ⟩ = P_{ℓχ}|φ_χ⟩ in the biorthonormal gauge P_{ℓχ}†P_{ℓψ} = 1, each bond update solves

    P_{ℓχ}† H P_{ℓψ} |φ_ψ⟩ = E |φ_ψ⟩,     P_{ℓψ}† H† P_{ℓχ} |φ_χ⟩ = Ē |φ_χ⟩,

normalizes ⟨φ_χ|φ_ψ⟩ = 1, and refactorizes both blocks by an oblique projection.
"""
module BiorthoDMRG

using ITensors: ITensors, ITensor, Index, array, commonind, dag, dim, hasqns, hassameinds,
    itensor, noprime, prime, replaceind, scalar, sim, swapprime, uniqueinds, onehot
using ITensorMPS: ITensorMPS, MPO, MPS, Sweeps, inner, maxlinkdim, siteind
using KrylovKit: EigSorter, eigsolve
using LinearAlgebra: LinearAlgebra, Diagonal, norm, rmul!
using Printf: @printf

export AbstractBiorthoProjMPO, BiorthoObserver, BiorthoProjMPO, BiorthoProjMPO_MPS,
    BiorthoProjMPS, adjoint_mpo, adjoint_product, biorthodmrg, biorthogonalize!,
    biorthoproblemsolver, biorthoreplacebond!, condition_number, residuals

include("environments.jl")
include("projmpo.jl")
include("projmps.jl")
include("projmpo_mps.jl")
include("factorization.jl")
include("gauge.jl")
include("solver.jl")
include("diagnostics.jl")
include("observer.jl")
include("bodmrg.jl")
include("models.jl")

end
