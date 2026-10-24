"""
    BiorthoProjSum(terms, coeffs)

`Σₐ coeffs[a] * terms[a]`, a linear combination of projected operators on the same
window. The adjoint conjugates the coefficients.
"""
struct BiorthoProjSum{T <: Number} <: AbstractBiorthoProjMPO
    terms::Vector{AbstractBiorthoProjMPO}
    coeffs::Vector{T}
    function BiorthoProjSum(
            terms::Vector{<:AbstractBiorthoProjMPO}, coeffs::Vector{T}
        ) where {T <: Number}
        isempty(terms) && throw(ArgumentError("a sum needs at least one term"))
        length(coeffs) == length(terms) || throw(DimensionMismatch("one coefficient per term"))
        allequal(length.(terms)) || throw(DimensionMismatch("the terms have different lengths"))
        return new{T}(terms, coeffs)
    end
end

Base.length(P::BiorthoProjSum) = length(first(P.terms))

function ITensorMPS.position!(P::BiorthoProjSum, chi::MPS, psi::MPS, ℓ::Int)
    foreach(t -> ITensorMPS.position!(t, chi, psi, ℓ), P.terms)
    return P
end

function ITensors.product(P::BiorthoProjSum, v::ITensor)
    return sum(c * ITensors.product(t, v) for (c, t) in zip(P.coeffs, P.terms))
end

function adjoint_product(P::BiorthoProjSum, v::ITensor)
    return sum(conj(c) * adjoint_product(t, v) for (c, t) in zip(P.coeffs, P.terms))
end

"""
    BiorthoProjMPO_MPS(H::MPO, refs::Vector{Tuple{MPS,MPS}}; weight)

`H` plus deflation terms for converged eigenpairs: the `BiorthoProjSum` of
`BiorthoProjMPO(H)` with coefficient `1` and one `BiorthoProjMPS(L_i, R_i)` per pair
with coefficient `weight`. Each pair is `(L_i, R_i) = (chi_i, psi_i)`, in the order
`biorthodmrg` returns them. For exact pairs with nondegenerate `E_i`, each `E_i` moves
to `E_i + weight` and every other eigenpair is unchanged, so choose `weight` to move
the converged eigenvalues out of the region selected by `which`.
"""
function BiorthoProjMPO_MPS(H::MPO, refs::Vector{<:Tuple{MPS, MPS}}; weight::Number)
    N = length(H)
    all(((L, R),) -> length(L) == length(R) == N, refs) ||
        throw(DimensionMismatch("every reference must have the length of H"))
    terms = AbstractBiorthoProjMPO[BiorthoProjMPO(H); [BiorthoProjMPS(L, R) for (L, R) in refs]]
    return BiorthoProjSum(terms, [one(weight); fill(weight, length(refs))])
end