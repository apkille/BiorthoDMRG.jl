using Test
using Random: Random
using ITensors, ITensorMPS
using BiorthoDMRG
using BiorthoDMRG.Models: xxz

Random.seed!(1234)

@testset "operators" begin
    N, Δ, g, h = 8, 0.5, 0.3, 0.1
    sites = siteinds("S=1/2", N)
    H = xxz(sites; Δ, g, h)

    @testset "adjoint_mpo: ⟨x|H†|y⟩ = conj ⟨y|H|x⟩" begin
        x, y = random_mps(ComplexF64, sites; linkdims = 3), random_mps(ComplexF64, sites; linkdims = 3)
        @test inner(x', adjoint_mpo(H), y) ≈ conj(inner(y', H, x))
    end

    @testset "adjoint_product: ⟨w|P v⟩ = ⟨P† w|v⟩ at every bond" begin
        chi, psi = random_mps(ComplexF64, sites; linkdims = 4), random_mps(ComplexF64, sites; linkdims = 4)
        biorthogonalize!(chi, psi, 1)
        refs = [(random_mps(ComplexF64, sites; linkdims = 2), random_mps(ComplexF64, sites; linkdims = 2))]
        P = BiorthoProjMPO_MPS(H, refs; weight = 2.0 - 1.0im)
        for ℓ in 1:(N - 1)
            ITensorMPS.position!(P, chi, psi, ℓ)
            window = inds(psi[ℓ] * psi[ℓ + 1])
            v, w = random_itensor(ComplexF64, window...), random_itensor(ComplexF64, window...)
            @test scalar(dag(w) * P(v)) ≈ scalar(dag(adjoint_product(P, w)) * v)
        end
    end
end