using Test
using LinearAlgebra: I
using Random: Random
using ITensors, ITensorMPS
using BiorthoDMRG

Random.seed!(1234)

# Contracting dag(chi[k]) with psi[k] over every index except `l` gives the identity on `l`.
function isbiorthonormal(chi::MPS, psi::MPS, k::Int, l::Index)
    return Array(dag(prime(chi[k], l)) * psi[k], l', l) ≈ Matrix(I, dim(l), dim(l))
end

@testset "biorthogonalize!: preserves ⟨χ|ψ⟩ and fixes the gauge" begin
    N = 8
    sites = siteinds("S=1/2", N)
    chi, psi = random_mps(sites; linkdims = 4), random_mps(sites; linkdims = 4)
    ov = inner(chi, psi)
    biorthogonalize!(chi, psi, 4; normalize = false)
    @test inner(chi, psi) ≈ ov rtol = 1.0e-8
    @test all(k -> isbiorthonormal(chi, psi, k, linkind(psi, k)), 1:3)
    @test all(k -> isbiorthonormal(chi, psi, k, linkind(psi, k - 1)), 5:N)
end