using Test
using LinearAlgebra: I, dot, eigen, eigvals, norm
using Random: Random
using ITensors, ITensorMPS
using BiorthoDMRG
using BiorthoDMRG.Models: doubled_siteinds, lindbladian, vectorized_identity, xxz

include("utils.jl")

Random.seed!(1234)

@testset "Lindbladian" begin
    Δ = 0.5
    kw = (; nsweeps = 10, maxdim = [10, 20, 40], cutoff = 1.0e-16, outputlevel = 0, eigsolve_krylovdim = 8)

    @testset "single qubit with amplitude damping" begin
        γ = 0.7
        s = siteinds("S=1/2", 1)
        d = doubled_siteinds(s)
        𝓛 = lindbladian(0 * MPO(s, "Id"), [sqrt(γ) * op("S-", s[1])], d)
        @test sort(real(eigvals(dense_matrix(𝓛, d)))) ≈ [-γ, -γ / 2, -γ / 2, 0] atol = 1.0e-12

        λ, chi, rho = biorthodmrg(
            𝓛, random_mps(ComplexF64, d; linkdims = 2);
            which = :LR, merge(kw, (; eigsolve_krylovdim = 4))...,
        )
        one = vectorized_identity(d)
        @test abs(λ) ≤ 1.0e-10
        @test inner(one, apply(op("Sz", s[1]), rho)) / inner(one, rho) ≈ -1 / 2
        @test abs(inner(chi, one)) ≈ nrm(chi) * nrm(one) rtol = 1.0e-8
    end

    @testset "vectorization: 𝓛|ρ⟩⟩ = |−i[H, ρ] + D(ρ)⟩⟩ for N = 2" begin
        ε = 0.5
        s = siteinds("S=1/2", 2)
        d = doubled_siteinds(s)
        H = xxz(s; Δ)
        𝓛 = lindbladian(H, [sqrt(ε) * op("S+", s[1]), sqrt(ε) * op("S-", s[2])], d)

        # ρ as a 4×4 matrix: rows (s₁, s₂), columns (b₁, b₂), first site fastest.
        as_matrix(x) = reshape(Array(x[1] * x[2] * x[3] * x[4], d[1], d[3], d[2], d[4]), 4, 4)
        id2 = Matrix(I, 2, 2)
        Hm = dense_matrix(H, s)
        Ls = [
            sqrt(ε) * kron(id2, Array(op("S+", s[1]), s[1]', s[1])),
            sqrt(ε) * kron(Array(op("S-", s[2]), s[2]', s[2]), id2),
        ]
        ρ = random_mps(ComplexF64, d; linkdims = 2)
        P = as_matrix(ρ)
        expected = -im * (Hm * P - P * Hm) + sum(L * P * L' - (L' * L * P + P * L' * L) / 2 for L in Ls)
        @test as_matrix(apply(𝓛, ρ)) ≈ expected rtol = 1.0e-10
    end

    @testset "driven XXZ: trace preservation and steady state" begin
        ε = 0.5
        s = siteinds("S=1/2", 4)
        d = doubled_siteinds(s)
        𝓛 = lindbladian(xxz(s; Δ), [sqrt(ε) * op("S+", s[1]), sqrt(ε) * op("S-", s[4])], d)
        one = vectorized_identity(d)
        @test abs(inner(one', 𝓛, random_mps(ComplexF64, d; linkdims = 4))) ≤ 1.0e-10

        F = eigen(dense_matrix(𝓛, d))
        i = argmax(real.(F.values))
        λ, chi, rho = biorthodmrg(𝓛, random_mps(ComplexF64, d; linkdims = 4); which = :LR, kw...)
        v, r = F.vectors[:, i], dense_vector(rho, d)
        @test abs(F.values[i]) ≤ 1.0e-10
        @test abs(λ) ≤ 1.0e-8
        @test abs(dot(v, r)) ≈ norm(v) * norm(r) rtol = 1.0e-6
        @test abs(inner(chi, one)) ≈ nrm(chi) * nrm(one) rtol = 1.0e-6
    end
end