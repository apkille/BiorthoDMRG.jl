const ROOT = 0

# The other ranks wait on this one, in a collective or a message exchange, so an error here
# cannot be recovered and aborts the communicator. With one rank nobody waits, and the error
# is rethrown.
function _abortonerror(f, comm::MPI.Comm)
    try
        return f()
    catch err
        MPI.Comm_size(comm) == 1 && rethrow()
        @error "rank $(MPI.Comm_rank(comm)) failed while the other ranks wait on it" exception = (err, catch_backtrace())
        MPI.Abort(comm, 1)
    end
end

# `obj` of every rank, in rank order, on rank 0, and `nothing` elsewhere. The objects travel
# by serialization, so they need not be isbits.
function _gatherobjects(obj, comm::MPI.Comm)
    MPI.Comm_rank(comm) == ROOT || (MPI.send(obj, comm; dest = ROOT); return nothing)
    return [p == ROOT ? obj : MPI.recv(comm; source = p) for p in 0:(MPI.Comm_size(comm) - 1)]
end