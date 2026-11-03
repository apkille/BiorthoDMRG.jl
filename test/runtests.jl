using Test
using BiorthoDMRG: BiorthoDMRG
using MPI: mpiexec

const TESTFILES = ["test_operators.jl", "test_gauge.jl", "test_solver.jl", "test_dmrg.jl", "test_lindblad.jl"]

# Ranks for test_mpi.jl, which accepts at most 9.
const NRANKS = 3

@testset "BiorthoDMRG" begin
    for file in TESTFILES
        @eval module $(Symbol(chopsuffix(file, ".jl")))
            include($(joinpath(@__DIR__, file)))
        end
    end

    # BiorthoDMRG and MPI are loaded above, so the extension is compiled before the
    # ranks start, and the ranks do not race to precompile it.
    @testset "MPI, $NRANKS ranks" begin
        cmd = `$(mpiexec()) -n $NRANKS $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(joinpath(@__DIR__, "test_mpi.jl"))`
        @test success(run(ignorestatus(cmd)))
    end
end