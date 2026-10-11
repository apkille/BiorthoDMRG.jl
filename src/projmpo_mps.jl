"""
    BiorthoProjMPO_MPS(H::MPO, refs::Vector{Tuple{MPS,MPS}}; weight)

`H` plus deflation terms for converged eigenpairs, projected onto the open
window. Use it to target the next eigenvalue once the pairs in `refs` are known.

`PH` is the projection of `H`. `pms` holds one `BiorthoProjMPS` per pair, the
projection of `|R_i⟩⟨L_i|`. `weight` multiplies every deflation term, so
`P(v) = PH(v) + weight * Σ_i pms[i](v)`.

`BiorthoProjMPO_MPS(H, refs; weight)` builds `PH` and one deflation term per pair.
Each pair is `(L_i, R_i) = (chi_i, psi_i)`, in the order `biorthodmrg` returns
them. For exact pairs, each `E_i` moves to `E_i + weight` and every other
eigenpair is unchanged, so choose `weight` to move the converged eigenvalues out
of the region selected by `which`.
"""
struct BiorthoProjMPO_MPS{W <: Number} <: AbstractBiorthoProjMPO
    PH::BiorthoProjMPO
    pms::Vector{BiorthoProjMPS}
    weight::W
end

function BiorthoProjMPO_MPS(H::MPO, refs::Vector{<:Tuple{MPS, MPS}}; weight::Number)
    N = length(H)
    all(((L, R),) -> length(L) == length(R) == N, refs) ||
        throw(DimensionMismatch("every reference must have the length of H"))
    return BiorthoProjMPO_MPS(BiorthoProjMPO(H), [BiorthoProjMPS(L, R) for (L, R) in refs], weight)
end

Base.length(P::BiorthoProjMPO_MPS) = length(P.PH)

function ITensorMPS.position!(P::BiorthoProjMPO_MPS, chi::MPS, psi::MPS, ℓ::Int)
    ITensorMPS.position!(P.PH, chi, psi, ℓ)
    foreach(pm -> ITensorMPS.position!(pm, chi, psi, ℓ), P.pms)
    return P
end

function ITensors.product(P::BiorthoProjMPO_MPS, v::ITensor)
    return sum((P.weight * ITensors.product(pm, v) for pm in P.pms); init = ITensors.product(P.PH, v))
end

function adjoint_product(P::BiorthoProjMPO_MPS, v::ITensor)
    return sum((conj(P.weight) * adjoint_product(pm, v) for pm in P.pms); init = adjoint_product(P.PH, v))
end