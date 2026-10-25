using MPI, ITensors, ITensorMPS, BiorthoDMRG, BenchmarkTools
using LinearAlgebra: BLAS

MPI.Init()
comm = MPI.COMM_WORLD
rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
BLAS.set_num_threads(1)

Lx, Ly, Δ, g = 4, 8, 1.5, 0.1
maxdim, nsweeps = 64, 2
nranks ≤ Ly || error("this partition needs at most Ly = $Ly ranks")

# The bonds of the cylinder whose earlier site, in the column-by-column order, lies in `rows`.
function cylinder(sites; rows = 1:Ly)
    site(x, y) = (x - 1) * Ly + y
    os = OpSum()
    for x in 1:Lx, y in rows
        i = site(x, y)
        later = Tuple{Int, Float64}[]
        y < Ly && push!(later, (site(x, y + 1), 0.0))
        y == 1 && push!(later, (site(x, Ly), 0.0))
        x < Lx && push!(later, (site(x + 1, y), g))
        for (j, a) in later
            os += exp(a) / 2, "S+", j, "S-", i
            os += exp(-a) / 2, "S-", j, "S+", i
            os += Δ, "Sz", i, "Sz", j
        end
    end
    return MPO(os, sites)
end

if rank == 0
    sites = siteinds("S=1/2", Lx * Ly)
    psi0 = random_mps(sites; linkdims = maxdim)
end
sites, psi0 = MPI.bcast(rank == 0 ? (sites, psi0) : nothing, comm; root = 0)
Hp = cylinder(sites; rows = (rank + 1):nranks:Ly)

# A new operator for every run, since a used one carries stale environments.
fresh() = MPIBiorthoSum(BiorthoProjMPO(Hp), comm; mode = :root)

# One call of `biorthodmrg`. Rank 0 sweeps, and the other ranks apply their parts of H.
function solve(PH)
    rank == 0 || return serve!(PH)
    try
        biorthodmrg(PH, psi0, psi0; nsweeps, maxdim, cutoff = 0.0, outputlevel = 0)
    finally
        stopservice!(PH)
    end
end

# Fixed samples and no time limit, so that every rank takes the same number of runs.
t = @belapsed solve(PH) setup = (PH = fresh()) evals = 1 samples = 3 seconds = Inf
rank == 0 && write(joinpath(@__DIR__, "time_$nranks.txt"), string(t))