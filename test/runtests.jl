using Test
using LinearAlgebra: I, eigvals, norm
using Random: Random
using ITensors, ITensorMPS
using BiorthoDMRG
using BiorthoDMRG.Models: ising_imaginary_field, nonreciprocal_ssh, xxz

Random.seed!(1234)

# Contracting dag(chi[k]) with psi[k] over every index except `l` gives the identity on `l`.
function isbiorthonormal(chi::MPS, psi::MPS, k::Int, l::Index)
    return Array(dag(prime(chi[k], l)) * psi[k], l', l) ≈ Matrix(I, dim(l), dim(l))
end

# Dense matrix of a two-site MPO, with primed indices as rows.
twosite_matrix(H::MPO, s) = reshape(Array(H[1] * H[2], s[1]', s[2]', s[1], s[2]), 4, 4)

@testset "BiorthogonalDMRG" begin
    N, Δ, g, h = 8, 0.5, 0.3, 0.1
    sites = siteinds("S=1/2", N)
    psi0 = random_mps(sites; linkdims = 4)
    kw = (; nsweeps = 10, maxdim = [10, 20, 40], cutoff = 1.0e-12, outputlevel = 0, eigsolve_krylovdim = 8)

    @testset "operators" begin
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

    @testset "biorthogonalize!: preserves ⟨χ|ψ⟩ and fixes the gauge" begin
        chi, psi = random_mps(sites; linkdims = 4), random_mps(sites; linkdims = 4)
        ov = inner(chi, psi)
        biorthogonalize!(chi, psi, 4; normalize = false)
        @test inner(chi, psi) ≈ ov rtol = 1.0e-8
        @test all(k -> isbiorthonormal(chi, psi, k, linkind(psi, k)), 1:3)
        @test all(k -> isbiorthonormal(chi, psi, k, linkind(psi, k - 1)), 5:N)
    end

    @testset "local solver: normalization and both eigen-equations" begin
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

    @testset "limits" begin
        @testset "N = 2 is exact diagonalization" begin
            s = siteinds("S=1/2", 2)
            H = xxz(s; Δ, g)
            E, = biorthodmrg(H, random_mps(s; linkdims = 2); merge(kw, (; eigsolve_krylovdim = 4))...)
            @test real(E) ≈ minimum(real, eigvals(twosite_matrix(H, s))) rtol = 1.0e-10
        end

        @testset "Hermitian limit (g = 0)" begin
            H = xxz(sites; Δ, g = 0.0, h)
            E, = biorthodmrg(H, psi0; kw...)
            E0, = dmrg(H, psi0; kw...)
            @test E ≈ E0 rtol = 1.0e-8
        end

        @testset "XXZ similarity (g ≠ 0): energy, residual stopping, κ ≥ 1" begin
            E0, = dmrg(xxz(sites; Δ, g = 0.0, h), psi0; kw...)
            H = xxz(sites; Δ, g, h)
            obs = BiorthoObserver(; H, residual_tol = 1.0e-6)
            E, chi, psi = biorthodmrg(H, psi0; observer = obs, kw...)
            @test E ≈ E0 rtol = 1.0e-8
            @test inner(chi, psi) ≈ 1 rtol = 1.0e-6
            @test maximum(last(obs.residuals)) < 1.0e-6
            @test all(≥(1 - 1.0e-10), obs.conditions)
        end

        @testset "SSH similarity: γ ≠ 0 matches t1′ = sign(t1) √(t1² − γ²/4)" begin
            t1, t2, γ, V = 1.0, 0.5, 0.6, 0.3
            f = siteinds("Fermion", N)
            phi0 = random_mps(f; linkdims = 4)
            t1′ = sign(t1) * sqrt(t1^2 - γ^2 / 4)
            E0, = dmrg(nonreciprocal_ssh(f; t1 = t1′, t2, γ = 0.0, V), phi0; kw...)
            E, = biorthodmrg(nonreciprocal_ssh(f; t1, t2, γ, V), phi0; kw...)
            @test E ≈ E0 rtol = 1.0e-8
        end

        @testset "complex-symmetric H: χ ∝ conj(ψ)" begin
            H = ising_imaginary_field(sites; hx = 1.5, hz = 0.2)
            E, chi, psi = biorthodmrg(H, psi0; kw...)
            @test abs(inner(dag(psi), chi)) ≈ norm(psi) * norm(chi) rtol = 1.0e-6
        end
    end

    @testset "deflation: shifted spectrum and ⟨L_0|ψ_1⟩ = 0" begin
        H0, H = xxz(sites; Δ, g = 0.0, h), xxz(sites; Δ, g, h)
        _, gs = dmrg(H0, psi0; kw...)
        E1h, = dmrg(H0, [gs], psi0; weight = 10.0, kw...)
        _, chi, psi = biorthodmrg(H, psi0; kw...)
        E1, _, psi1 = biorthodmrg(H, [(chi, psi)], psi0, psi0; weight = 10.0, kw...)
        @test E1 ≈ E1h rtol = 1.0e-6
        @test abs(inner(chi, psi1)) ≤ 1.0e-6 * norm(chi) * norm(psi1)
    end
end