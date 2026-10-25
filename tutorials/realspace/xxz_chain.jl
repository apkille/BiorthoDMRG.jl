using MPI, ITensors, ITensorMPS, BiorthoDMRG
using BiorthoDMRG.Models: xxz
using LinearAlgebra: BLAS

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
BLAS.set_num_threads(1)

N, Δ, g = 64, 1.5, 0.2
maxdim, nsweeps, samples = 64, 4, 2

# Sites, H, and an initial state on `n` sites, built on rank 0 and broadcast, so that every
# rank holds the same index ids.
function instance(n::Int, linkdims::Int)
    if rank == 0
        sites = siteinds("S=1/2", n)
        H, psi0 = xxz(sites; Δ, g), random_mps(sites; linkdims)
    end
    return MPI.bcast(rank == 0 ? (sites, H, psi0) : nothing, comm; root = 0)
end

# One call of `biorthodmrg` on every rank, timed from a common start. Rank 0 returns last,
# after joining the parts, so its time is that of the whole run. A new operator for every
# run, since a used one carries stale environments.
function timedsolve(H, psi0; nsweeps, maxdim)
    R = MPIRealSpace(BiorthoProjMPO(H), comm)
    MPI.Barrier(comm)
    t = MPI.Wtime()
    result = biorthodmrg(R, psi0; nsweeps, maxdim, cutoff = 0.0, outputlevel = 0)
    return MPI.Wtime() - t, result
end

# A small instance compiles the same methods, so the timed runs exclude compilation.
_, Hs, psis = instance(4 * nranks, 4)
timedsolve(Hs, psis; nsweeps = 1, maxdim = 4)

sites, H, psi0 = instance(N, maxdim)
runs = [timedsolve(H, psi0; nsweeps, maxdim) for _ in 1:samples]

# H is similar to the Hermitian chain at g = 0, so its lowest eigenvalue is the ground-state
# energy E0 of that chain. Serial `dmrg` computes it in the run on one rank, which the README
# driver starts first, and the runs on more ranks read it.
if rank == 0
    reference = joinpath(@__DIR__, "E0.txt")
    if nranks == 1
        E0, _ = dmrg(xxz(sites; Δ), psi0; nsweeps = 4 * nsweeps, maxdim, cutoff = 0.0, outputlevel = 0)
        write(reference, string(E0))
    end
    E0 = parse(Float64, read(reference, String))
    _, (E, _, _) = last(runs)
    write(joinpath(@__DIR__, "time_$nranks.txt"), string(minimum(first, runs)))
    write(joinpath(@__DIR__, "error_$nranks.txt"), string(abs(E - E0) / abs(E0)))
end