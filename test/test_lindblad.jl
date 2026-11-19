using Test
using LinearAlgebra: I, dot, eigen, eigvals, norm, tr
using Random: Random
using ITensors, ITensorMPS
using BiorthoDMRG
using BiorthoDMRG.Models: lindbladian, vectorized_identity, vectorized_siteinds, xxz

include("utils.jl")

# tr(A ρ) for an MPO `A` on the physical sites and `rho` on the vectorized sites `v`
trace_with(A, rho, v) = inner(vectorize(adjoint_mpo(A), v), rho)

# ρ as a dense matrix: rows the ket indices, columns the bra indices, first site fastest.
# Each vⱼ is (row, column) with the row fastest, so it splits into dimensions (dⱼ, dⱼ).
function as_matrix(x::MPS, v)
    d, N = isqrt.(dim.(v)), length(v)
    A = reshape(Array(prod(x[k] for k in 1:N), v...), Iterators.flatten((dk, dk) for dk in d)...)
    return reshape(permutedims(A, [1:2:(2N); 2:2:(2N)]), prod(d), prod(d))
end

Random.seed!(1234)

@testset "Lindbladian" begin
    Δ = 0.5
    kw = (; nsweeps = 10, maxdim = [10, 20, 40], cutoff = 1.0e-16, outputlevel = 0, eigsolve_krylovdim = 8)
    # mindim > 1 keeps every block out of the partial-trace reduction of the test map
    sskw = (; nsweeps = 10, maxdim = [10, 20, 40], mindim = 4, cutoff = 1.0e-16, outputlevel = 0)

    @testset "two qubits with amplitude damping" begin
        γ = 0.7
        s = siteinds("S=1/2", 2)
        v = vectorized_siteinds(s)
        𝓛 = lindbladian(0 * MPO(s, "Id"), [sqrt(γ) * op("S-", s[1]), sqrt(γ) * op("S-", s[2])], v)
        e1 = [-γ, -γ / 2, -γ / 2, 0]
        @test sort(real(eigvals(dense_matrix(𝓛, v)))) ≈ sort(vec([a + b for a in e1, b in e1])) atol = 1.0e-12

        λ, chi, rho = biorthodmrg(𝓛, random_mps(ComplexF64, v; linkdims = 2); which = :LR, kw...)
        one = vectorized_identity(v)
        @test abs(λ) ≤ 1.0e-10
        @test trace_with(MPO(s, ["Sz", "Id"]), rho, v) / inner(one, rho) ≈ -1 / 2
        @test abs(inner(chi, one)) ≈ nrm(chi) * nrm(one) rtol = 1.0e-8
    end

    @testset "vectorize: the identity and tr(Aρ) against dense matrices" begin
        s = siteinds("S=1/2", 2)
        v = vectorized_siteinds(s)
        rho = random_mps(ComplexF64, v; linkdims = 2)
        @test norm(dense_vector(vectorize(MPO(s, "Id"), v), v) - dense_vector(vectorized_identity(v), v)) ≤ 1.0e-14
        A = MPO(OpSum() + ("S+", 1, "S-", 2) + (0.3im, "Sz", 1), s)
        @test trace_with(A, rho, v) ≈ tr(dense_matrix(A, s) * as_matrix(rho, v))
    end

    @testset "vectorization: 𝓛|ρ⟩⟩ = |−i[H, ρ] + D(ρ)⟩⟩ for N = 2" begin
        ε = 0.5
        s = siteinds("S=1/2", 2)
        v = vectorized_siteinds(s)
        H = xxz(s; Δ)
        𝓛 = lindbladian(H, [sqrt(ε) * op("S+", s[1]), sqrt(ε) * op("S-", s[2])], v)

        id2 = Matrix(I, 2, 2)
        Hm = dense_matrix(H, s)
        Ls = [
            sqrt(ε) * kron(id2, Array(op("S+", s[1]), s[1]', s[1])),
            sqrt(ε) * kron(Array(op("S-", s[2]), s[2]', s[2]), id2),
        ]
        ρ = random_mps(ComplexF64, v; linkdims = 2)
        P = as_matrix(ρ, v)
        expected = -im * (Hm * P - P * Hm) + sum(L * P * L' - (L' * L * P + P * L' * L) / 2 for L in Ls)
        @test as_matrix(apply(𝓛, ρ), v) ≈ expected rtol = 1.0e-10
    end

    @testset "driven XXZ: trace preservation and steady state" begin
        ε = 0.5
        s = siteinds("S=1/2", 4)
        v = vectorized_siteinds(s)
        𝓛 = lindbladian(xxz(s; Δ), [sqrt(ε) * op("S+", s[1]), sqrt(ε) * op("S-", s[4])], v)
        one = vectorized_identity(v)
        @test abs(inner(one', 𝓛, random_mps(ComplexF64, v; linkdims = 4))) ≤ 1.0e-10

        F = eigen(dense_matrix(𝓛, v))
        i = argmax(real.(F.values))
        λ, chi, rho = biorthodmrg(𝓛, random_mps(ComplexF64, v; linkdims = 4); which = :LR, kw...)
        w, r = F.vectors[:, i], dense_vector(rho, v)
        @test abs(F.values[i]) ≤ 1.0e-10
        @test abs(λ) ≤ 1.0e-8
        @test abs(dot(w, r)) ≈ norm(w) * norm(r) rtol = 1.0e-6
        @test abs(inner(chi, one)) ≈ nrm(chi) * nrm(one) rtol = 1.0e-6
    end

    @testset "LindbladProjMPO: P_{ℓ1}†P_{ℓρ} = 1 and ⟨φ₁| is a left null vector" begin
        ε = 0.5
        s = siteinds("S=1/2", 3)
        v = vectorized_siteinds(s)
        id = vectorized_identity(v)
        𝓛 = lindbladian(xxz(s; Δ), [sqrt(ε) * op("S+", s[1]), sqrt(ε) * op("S-", s[3])], v)
        P = LindbladProjMPO(𝓛)
        P1 = LindbladProjMPO(MPO(v, "Id"))
        @test all(k -> siteind(P.id, k) == v[k], eachindex(v))
        rho = random_mps(ComplexF64, v; linkdims = 4)
        for ℓ in 1:(length(v) - 1)
            orthogonalize!(rho, ℓ)
            ITensorMPS.position!(P, rho, ℓ)
            ITensorMPS.position!(P1, rho, ℓ)
            phi = rho[ℓ] * rho[ℓ + 1]
            x = random_itensor(ComplexF64, inds(phi))
            @test P1(x) ≈ x
            @test abs(scalar(P.bra1 * P(x))) ≤ 1.0e-10 * norm(P.bra1) * norm(P(x))
            @test scalar(P.bra1 * phi) ≈ inner(id, rho)
        end
        @test_throws BoundsError ITensorMPS.position!(P, rho, length(v))
    end

    @testset "steadystate: single qubit with amplitude damping" begin
        γ = 0.7
        s = siteinds("S=1/2", 1)
        v = vectorized_siteinds(s)
        id = vectorized_identity(v)
        𝓛 = lindbladian(0 * MPO(s, "Id"), [sqrt(γ) * op("S-", s[1])], v)
        λ, rho = steadystate(𝓛; sskw...)
        @test abs(λ) ≤ 1.0e-10
        @test inner(id, rho) ≈ 1
        @test trace_with(MPO(s, ["Sz"]), rho, v) ≈ -1 / 2
    end

    @testset "steadystate: driven XXZ against exact diagonalization" begin
        ε = 0.5
        s = siteinds("S=1/2", 4)
        v = vectorized_siteinds(s)
        id = vectorized_identity(v)
        𝓛 = lindbladian(xxz(s; Δ), [sqrt(ε) * op("S+", s[1]), sqrt(ε) * op("S-", s[4])], v)
        F = eigen(dense_matrix(𝓛, v))
        w = F.vectors[:, argmax(real.(F.values))]
        obs = BiorthoObserver(; H = 𝓛)
        λ, rho = steadystate(𝓛; observer = obs, sskw...)
        r = dense_vector(rho, v)
        @test abs(λ) ≤ 1.0e-8
        @test inner(id, rho) ≈ 1
        @test abs(dot(w, r)) ≈ norm(w) * norm(r) rtol = 1.0e-6
        @test first(last(obs.residuals)) ≤ 1.0e-6

        # a correlated initial state reaches the same steady state
        _, rho′ = steadystate(𝓛, random_mps(ComplexF64, v; linkdims = 4); sskw...)
        r′ = dense_vector(rho′, v)
        @test abs(dot(w, r′)) ≈ norm(w) * norm(r′) rtol = 1.0e-6
    end
end