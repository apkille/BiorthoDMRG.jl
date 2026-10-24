# mpiexecjl -n 3 julia --project test/test_mpi.jl
using MPI, ITensors, ITensorMPS, BiorthoDMRG, Test
using BiorthoDMRG.Models: ising_imaginary_field, xxz
using LinearAlgebra: norm

MPI.Init()
const comm = MPI.COMM_WORLD
const rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
const N = 10
nranks ≤ N - 1 || error("run with at most $(N - 1) ranks")

# Rank 0 holds the imaginary field of `Models.ising_imaginary_field` and the other ranks
# share its real terms, so the ranks' MPOs differ in entry type.
function ising_part(sites; hx, hz)
    os = OpSum()
    rank == 0 && for j in eachindex(sites)
        os += -im * hz, "Z", j
    end
    if rank > 0 || nranks == 1
        r, n = nranks == 1 ? (0, 1) : (rank - 1, nranks - 1)
        for j in (r + 1):n:(length(sites) - 1)
            os += -1.0, "Z", j, "Z", j + 1
        end
        for j in (r + 1):n:length(sites)
            os += -hx, "X", j
        end
    end
    return MPO(os, sites)
end

function xxz_part(sites; Δ, g)
    os = OpSum()
    for j in (rank + 1):nranks:(length(sites) - 1)
        os += exp(g) / 2, "S+", j + 1, "S-", j
        os += exp(-g) / 2, "S-", j + 1, "S+", j
        os += Δ, "Sz", j, "Sz", j + 1
    end
    return MPO(os, sites)
end

function inputs()
    s = siteinds("S=1/2", N)
    chi, psi = biorthogonalize!(random_mps(s; linkdims = 4), random_mps(s; linkdims = 4), 1)
    return s, chi, psi
end
const sites, chi, psi = MPI.bcast(rank == 0 ? inputs() : nothing, comm; root = 0)
const hx, hz = 0.7, 0.3

# Relative deviations of P's products from those of `ref`, and of the adjoint identity
# ⟨w|P v⟩ = conj(⟨v|P† w⟩), at the window starting at ℓ.
function product_errors(P, ref, ℓ)
    ITensorMPS.position!(ref, chi, psi, ℓ)
    ITensorMPS.position!(P, chi, psi, ℓ)
    v, w = psi[ℓ] * psi[ℓ + 1], chi[ℓ] * chi[ℓ + 1]
    Pv, Pdw = ITensors.product(P, v), adjoint_product(P, w)
    return (
        norm(Pv - ITensors.product(ref, v)) / norm(Pv),
        norm(Pdw - adjoint_product(ref, w)) / norm(Pdw),
        abs(scalar(dag(w) * Pv) - conj(scalar(dag(v) * Pdw))) / (norm(w) * norm(Pv)),
    )
end

@testset "replicated products, mixed entry types" begin
    P = MPIBiorthoSum(BiorthoProjMPO(ising_part(sites; hx, hz)), comm; mode = :replicated, checksync = true)
    ref = BiorthoProjMPO(ising_imaginary_field(sites; hx, hz))
    for ℓ in 1:(N - 1)
        @test all(≤(1.0e-12), product_errors(P, ref, ℓ))
    end
end

@testset "root-driven products, mixed entry types" begin
    P = MPIBiorthoSum(BiorthoProjMPO(ising_part(sites; hx, hz)), comm; mode = :root)
    if rank == 0
        ref = BiorthoProjMPO(ising_imaginary_field(sites; hx, hz))
        try
            for ℓ in 1:(N - 1)
                @test all(≤(1.0e-12), product_errors(P, ref, ℓ))
            end
        finally
            stopservice!(P)
        end
    else
        serve!(P)
    end
end

@testset "spectrum of S H₀ S⁻¹ against Hermitian DMRG" begin
    Δ, g = 0.5, 0.1
    P = MPIBiorthoSum(BiorthoProjMPO(xxz_part(sites; Δ, g)), comm; mode = :root)
    if rank == 0
        E = try
            first(biorthodmrg(P, psi, psi; nsweeps = 10, maxdim = 32, cutoff = 1.0e-12, outputlevel = 0))
        finally
            stopservice!(P)
        end
        E0, _ = dmrg(xxz(sites; Δ, g = 0.0), psi; nsweeps = 10, maxdim = 32, cutoff = 1.0e-12, outputlevel = 0)
        @test abs(E - E0) ≤ 1.0e-8 * abs(E0)
    else
        serve!(P)
    end
end