function _commontype(T::Type{<:Number}, comm::MPI.Comm)
    MPI.Allreduce(Int(real(T) === Float64), min, comm) == 1 ||
        throw(ArgumentError("every rank's term must have Float64 or ComplexF64 entries"))
    return MPI.Allreduce(Int(T <: Complex), max, comm) == 1 ? ComplexF64 : Float64
end

function BiorthoDMRG.MPIBiorthoSum(
        term::AbstractBiorthoProjMPO, comm::MPI.Comm; mode::Symbol, checksync::Bool = false,
    )
    mode in (:root, :replicated) ||
        throw(ArgumentError("mode must be :root or :replicated, got :$mode"))
    T = _commontype(_scalartype(term), comm)
    return MPIBiorthoSum(term, comm, T, mode, MPI.Comm_rank(comm) == ROOT, checksync, SiteTracker())
end

BiorthoDMRG.MPIObserver(obs, comm::MPI.Comm) = MPIObserver(obs, comm, MPI.Comm_rank(comm) == ROOT)

ITensorMPS.checkdone!(o::MPIObserver; kwargs...) =
    MPI.bcast(o.isroot && ITensorMPS.checkdone!(o.obs; kwargs...), o.comm; root = ROOT)

_checkroot(P::MPIBiorthoSum) = P.isroot || error(
    "in mode :root only rank 0 calls position!, product, adjoint_product and stopservice!; " *
        "the other ranks call serve!"
)

# Σ over ranks of `w`, laid out in the index order of `v` and converted to the common
# entry type. Valid on rank 0, and on every rank in mode :replicated.
function _ranksum(P::MPIBiorthoSum, w::ITensor, v::ITensor)
    A = convert(Array{promote_type(eltype(w), P.scalartype)}, Array(w, inds(v)...))
    MPI.Reduce!(A, +, P.comm; root = ROOT)
    P.mode === :replicated && MPI.Bcast!(A, P.comm; root = ROOT)
    return itensor(A, inds(v)...)
end

# In mode :root, rank 0 asks the workers to apply their terms (`tag`) to `v`.
function _request(P::MPIBiorthoSum, tag::Symbol, v::ITensor)
    P.mode === :root || return nothing
    _checkroot(P)
    MPI.bcast((tag, v), P.comm; root = ROOT)
    return nothing
end

# One step of a splitmix64-style hash.
function _mix(h::UInt64, x::UInt64)
    z = h + x * 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

# Checksum of ℓ and of the indices and entries of the site tensors at `ks`.
function _checksum(chi::MPS, psi::MPS, ks::Vector{Int}, ℓ::Int)
    h = _mix(zero(UInt64), UInt64(ℓ))
    for k in ks, T in (chi[k], psi[k])
        h = _mix(h, UInt64(k))
        for i in inds(T)
            h = _mix(_mix(h, ITensors.id(i)), UInt64(ITensors.plev(i)))
        end
        for x in reinterpret(UInt64, vec(array(T)))
            h = _mix(h, x)
        end
    end
    return h
end

function ITensorMPS.position!(P::MPIBiorthoSum, chi::MPS, psi::MPS, ℓ::Int)
    P.mode === :root && _checkroot(P)
    ks = changedsites!(P.tracker, chi, psi)
    if P.mode === :root
        msg = (:position, ℓ, length(psi), ks, ITensor[chi[k] for k in ks], ITensor[psi[k] for k in ks])
        MPI.bcast(msg, P.comm; root = ROOT)
    elseif P.checksync
        h = _checksum(chi, psi, ks, ℓ)
        MPI.Allreduce(h, min, P.comm) == MPI.Allreduce(h, max, P.comm) ||
            error("the ranks hold different states at the window starting at site $ℓ")
    end
    ITensorMPS.position!(P.term, chi, psi, ℓ)
    return P
end

function ITensors.product(P::MPIBiorthoSum, v::ITensor)
    _request(P, :product, v)
    w = _abortonerror(P.comm) do
        ITensors.product(P.term, v)
    end
    return _ranksum(P, w, v)
end

function BiorthoDMRG.adjoint_product(P::MPIBiorthoSum, v::ITensor)
    _request(P, :adjoint, v)
    w = _abortonerror(P.comm) do
        adjoint_product(P.term, v)
    end
    return _ranksum(P, w, v)
end

function BiorthoDMRG.serve!(P::MPIBiorthoSum)
    (P.mode === :root && !P.isroot) ||
        error("serve! runs in mode :root, on the ranks other than 0")
    chi = psi = nothing
    try
        while true
            msg = MPI.bcast(nothing, P.comm; root = ROOT)
            tag = first(msg)
            tag === :stop && return P
            if tag === :position
                _, ℓ, N, ks, chik, psik = msg
                if isnothing(psi) || length(psi) != N
                    ks == 1:N || error("a new worker state needs every site")
                    chi, psi = MPS(chik), MPS(psik)
                else
                    for (k, c, s) in zip(ks, chik, psik)
                        chi[k], psi[k] = c, s
                    end
                end
                ITensorMPS.position!(P.term, chi, psi, ℓ)
            else
                v = last(msg)
                w = tag === :product ? ITensors.product(P.term, v) : adjoint_product(P.term, v)
                _ranksum(P, w, v)
            end
        end
    catch err
        @error "worker $(MPI.Comm_rank(P.comm)) failed" exception = (err, catch_backtrace())
        MPI.Abort(P.comm, 1)
    end
end

function BiorthoDMRG.stopservice!(P::MPIBiorthoSum)
    P.mode === :root || error("stopservice! applies to mode :root")
    _checkroot(P)
    MPI.bcast((:stop,), P.comm; root = ROOT)
    P.tracker.chi, P.tracker.psi = ITensor[], ITensor[]
    return P
end