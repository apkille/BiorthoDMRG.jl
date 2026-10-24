# mpiexecjl -n 4 julia --project examples/mpi_xxz.jl
using MPI, ITensors, ITensorMPS, BiorthoDMRG
using BiorthoDMRG.Models: xxz

MPI.Init()
comm = MPI.COMM_WORLD
rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
N, Δ, g = 40, 0.5, 0.1
nranks ≤ N - 1 || error("every rank needs at least one bond")

# The bonds `bonds` of `Models.xxz`; over j ≡ rank (mod nranks) the parts sum to it.
function xxz_part(sites, bonds; Δ, g)
    os = OpSum()
    for j in bonds
        os += exp(g) / 2, "S+", j + 1, "S-", j
        os += exp(-g) / 2, "S-", j + 1, "S+", j
        os += Δ, "Sz", j, "Sz", j + 1
    end
    return MPO(os, sites)
end

inputs() = (s = siteinds("S=1/2", N); (s, random_mps(s; linkdims = 16)))
sites, psi0 = MPI.bcast(rank == 0 ? inputs() : nothing, comm; root = 0)

# With deflation, rank 0 alone uses BiorthoProjMPO_MPS(H_0, refs; weight) as its term.
PH = MPIBiorthoSum(BiorthoProjMPO(xxz_part(sites, (rank + 1):nranks:(N - 1); Δ, g)), comm; mode = :root)
if rank == 0
    observer = BiorthoObserver(; H = xxz(sites; Δ, g), residual_tol = 1.0e-6)
    E, chi, psi = try
        biorthodmrg(PH, psi0, psi0; nsweeps = 10, maxdim = [16, 32, 64, 128], cutoff = 1.0e-10, observer)
    finally
        stopservice!(PH)
    end
else
    serve!(PH)
end