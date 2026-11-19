"""
    vectorized_siteinds(s)

Site indices for vectorized density matrices on the sites `s`, one per site. The index `vⱼ`
has dimension `dim(sⱼ)²` and carries the pair (row, column) of ρ on site `j`, with the row
fastest: `vⱼ = a + d(c - 1)` for row `a` and column `c`, `d = dim(sⱼ)`.
"""
vectorized_siteinds(s::Vector{<:Index}) = [Index(dim(x)^2, "Site,vec,n=$j") for (j, x) in enumerate(s)]

# The tensor with entries 1 at `v = a + d(c - 1)` for `s = a` and `b = c`, which joins a row
# index `s` and a column index `b` of dimension `d` into the vectorized index `v`.
function _fuser(s::Index, b::Index, v::Index)
    d = dim(s)
    (dim(b) == d && dim(v) == d^2) ||
        throw(DimensionMismatch("cannot join dimensions $(dim(s)) and $(dim(b)) into $(dim(v))"))
    F = zeros(d, d, d^2)
    for a in 1:d, c in 1:d
        F[a, c, a + d * (c - 1)] = 1
    end
    return itensor(F, s, b, v)
end

"""
    vectorized_identity(v)

The identity matrix as an MPS on the vectorized sites `v`, a product state with bond
dimension `1`. Its overlap with a vectorized density matrix is the trace, so
`inner(vectorized_identity(v), rho)` is `tr ρ`.
"""
function vectorized_identity(v::Vector{<:Index})
    T = map(v) do x
        d = isqrt(dim(x))
        d^2 == dim(x) || throw(ArgumentError("a vectorized site has dimension d², got $(dim(x))"))
        e = zeros(dim(x))
        for a in 1:d
            e[a + d * (a - 1)] = 1
        end
        return itensor(e, x)
    end
    return MPS(_chain(T))
end

"""
    vectorize(A::MPO, v)

The operator `A`, an MPO on physical sites, as an MPS on the vectorized sites `v`, with the
row of `A` on each site taken from its primed index. `inner(vectorize(adjoint_mpo(A), v), rho)`
is `tr(Aρ)` for a vectorized density matrix `rho`, and
`vectorize(MPO(s, "Id"), v)` is `vectorized_identity(v)`.
"""
function vectorize(A::MPO, v::Vector{<:Index})
    length(A) == length(v) || throw(DimensionMismatch("A and v have different lengths"))
    s = firstsiteinds(A; plev = 0)
    return MPS([A[j] * _fuser(s[j]', s[j], v[j]) for j in eachindex(v)])
end

# join consecutive tensors that share no index with a link of dimension 1
function _chain(T::Vector{ITensor})
    T = copy(T)
    for k in 1:(length(T) - 1)
        if !hascommoninds(T[k], T[k + 1])
            e = Index(1, "Link")
            T[k] *= onehot(e => 1)
            T[k + 1] *= onehot(e => 1)
        end
    end
    return T
end