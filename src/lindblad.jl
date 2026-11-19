"""
    lindbladian(H::MPO, jumps::Vector{ITensor}, v; cutoff = 1e-14)

The Lindbladian `ρ ↦ -i[H, ρ] + Σₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})` as an MPO acting on
vectorized density matrices on the sites `v = vectorized_siteinds(s)`. `H` is an MPO on
the physical sites `s`, and each jump operator is an ITensor on a single physical site,
for example `sqrt(γ) * op("S-", s[1])`. Each term is built on the ket and bra copies of
the sites, joined into `v` site by site, and the terms are summed and compressed with
`cutoff`.

The steady state is the right null vector with unit trace, and `vectorized_identity(v)`
is an exact left null vector, so `tr(Oρ)` is `inner(vectorize(adjoint_mpo(O), v), rho)`.
`steadystate(𝓛; nsweeps)` finds the steady state with the left state held fixed at the
identity.

A single-site `"C"` carries no Jordan–Wigner string, so fermionic jump operators should 
be written in a spin representation.
"""
function lindbladian(H::MPO, jumps::Vector{ITensor}, v::Vector{<:Index}; cutoff::Real = 1e-14)
    length(v) == length(H) ||
        throw(DimensionMismatch("v must hold one index for every site of H"))
    s = firstsiteinds(H; plev = 0)
    b = [addtags(sim(x), "bra") for x in s]
    d = collect(Iterators.flatten(zip(s, b)))
    terms = [-im * _embed(H, d, :ket), im * _embed(H, d, :bra), (_dissipator(L, d) for L in jumps)...]
    return +((_fuse(T, s, b, v) for T in terms)...; cutoff)
end

# The MPO `M` on the doubled sites `d = [s₁, b₁, s₂, b₂, …]` as an MPO on the vectorized
# sites `v`, joining each ket–bra pair `(sⱼ, bⱼ)` into `vⱼ` on the input and output sides.
function _fuse(M::MPO, s::Vector{<:Index}, b::Vector{<:Index}, v::Vector{<:Index})
    T = map(eachindex(v)) do j
        F = _fuser(s[j], b[j], v[j])
        return M[2j - 1] * M[2j] * F * prime(F)
    end
    return MPO(T)
end

# `A` on the doubled sites `d`: `A ⊗ 1` for `side = :ket`, `1 ⊗ Aᵀ` for `side = :bra`
# Each link of `A` passes through the identity on the other copy of its site
function _embed(A::MPO, d::Vector{<:Index}, side::Symbol)
    N = length(A)
    s, b = d[1:2:end], d[2:2:end]
    l = [commonind(A[j], A[j + 1]) for j in 1:(N - 1)]
    l̃ = sim.(l)
    T = Vector{ITensor}(undef, 2N)
    for j in 1:N
        Aj = side === :ket ? A[j] : replaceinds(A[j], (s[j], s[j]'), (b[j]', b[j]))
        j > 1 && (Aj = replaceind(Aj, l[j - 1], l̃[j - 1]))
        if side === :ket
            T[2j - 1] = Aj
            T[2j] = j < N ? op("Id", b[j]) * dense(delta(l[j], l̃[j])) : op("Id", b[j])
        else
            T[2j - 1] = j > 1 ? op("Id", s[j]) * dense(delta(l[j - 1], l̃[j - 1])) : op("Id", s[j])
            T[2j] = Aj
        end
    end
    return MPO(_chain(T))
end

# L ρ L† - ½{L†L, ρ} for a single-site jump L, as an MPO on the doubled sites
function _dissipator(L::ITensor, d::Vector{<:Index})
    s = only(inds(L; plev = 0))
    j = findfirst(==(s), d)
    (isnothing(j) || iseven(j)) && throw(ArgumentError("each jump operator must act on one physical site"))
    b = d[j + 1]
    LdL = replaceprime(prime(swapprime(dag(L), 0, 1)) * L, 2 => 1)
    c = Index(3, "Link")
    ket = L * onehot(c => 1) + LdL * onehot(c => 2) + op("Id", s) * onehot(c => 3)
    bra = replaceinds(dag(L), (s, s'), (b, b')) * onehot(c => 1) -
          op("Id", b) * onehot(c => 2) / 2 -
          replaceinds(LdL, (s, s'), (b', b)) * onehot(c => 3) / 2
    T = [op("Id", x) for x in d]
    T[j], T[j + 1] = ket, bra
    return MPO(_chain(T))
end