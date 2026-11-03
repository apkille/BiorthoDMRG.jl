using Test
using LinearAlgebra: norm
using Random: Random
using ITensors, ITensorMPS
using BiorthoDMRG
using BiorthoDMRG.Models: xxz

Random.seed!(1234)

@testset "local solver: normalization and both eigen-equations" begin
    Δ, g = 0.5, 0.3
    s = siteinds("S=1/2", 2)
    P = BiorthoProjMPO(xxz(s; Δ, g))
    chi, psi = biorthogonalize!(random_mps(s; linkdims = 2), random_mps(s; linkdims = 2), 1)
    ITensorMPS.position!(P, chi, psi, 1)
    E, x, y, _ = biorthoproblemsolver(P, psi[1] * psi[2], chi[1] * chi[2]; krylovdim = 4)
    @test scalar(dag(y) * x) ≈ 1
    @test norm(x) ≈ norm(y)
    @test norm(P(x) - E * x) ≤ 1.0e-10 * norm(x)
    @test norm(adjoint_product(P, y) - conj(E) * y) ≤ 1.0e-10 * norm(y)
end