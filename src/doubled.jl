"""
    doubled_siteinds(s)

Site indices for vectorized density matrices on the sites `s`, ordered
`[s₁, b₁, s₂, b₂, …]`. The physical index `sⱼ` carries the row (ket) index of ρ,
and `bⱼ`, a new index of the same site type tagged `"bra"`, carries the column
(bra) index.
"""
doubled_siteinds(s::Vector{<:Index}) = collect(Iterators.flatten((x, addtags(sim(x), "bra")) for x in s))

"""
    vectorized_identity(d)

The identity matrix as an MPS on the doubled sites `d`, ordered as by `doubled_siteinds`,
with bond dimension `dim(sⱼ)` inside each pair and `1` between pairs. Its overlap with a
vectorized density matrix is the trace, so `inner(vectorized_identity(d), rho)` is `tr ρ`.
"""
function vectorized_identity(d::Vector{<:Index})
    iseven(length(d)) || throw(ArgumentError("doubled sites come in ket–bra pairs"))
    T = ITensor[]
    for (s, b) in zip(d[1:2:end], d[2:2:end])
        c = Index(dim(s), "Link")
        push!(T, dense(delta(s, c)), dense(delta(c, b)))
    end
    return MPS(_chain(T))
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