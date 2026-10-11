"""
    biorthogonalize!(chi, psi, j; normalize=true, kwargs...)

Bring `chi` and `psi` into biorthonormal form centered on site `j`, in place, and
return them.

Each bond is refactorized with `biorthoreplacebond!`, sweeping in from both ends
toward `j`. Afterwards both states share every link index, sites left of `j` are
left-biorthonormal, and sites right of `j` are right-biorthonormal.
Left-biorthonormal means that contracting `dag(chi[k])` with `psi[k]` over their
left link and site index gives the identity; right-biorthonormal is the mirror
of this.

With `normalize=true`, `chi[j]` is rescaled so that `⟨chi|psi⟩ = 1`. `kwargs` are
passed to `biorthoreplacebond!`. Without truncation, the states are unchanged
when, at every bond, the kept bases of `chi` and `psi` have the same dimension
and no overlap cosine is dropped. Otherwise they are truncated.
"""
function biorthogonalize!(chi::MPS, psi::MPS, j::Int; normalize::Bool = true, kwargs...)
    N = length(psi)
    (length(chi) == N && all(k -> siteind(chi, k) == siteind(psi, k), 1:N)) ||
        throw(ArgumentError("chi and psi must share their site indices"))
    1 ≤ j ≤ N || throw(BoundsError(psi, j))
    for ℓ in 1:(j - 1)
        biorthoreplacebond!(
            chi, psi, ℓ, chi[ℓ] * chi[ℓ + 1], psi[ℓ] * psi[ℓ + 1]; 
            ortho = "left", kwargs...
        )
    end
    for ℓ in (N - 1):-1:j
        biorthoreplacebond!(
            chi, psi, ℓ, chi[ℓ] * chi[ℓ + 1], psi[ℓ] * psi[ℓ + 1]; 
            ortho = "right", kwargs...
        )
    end
    if normalize
        overlap = scalar(dag(chi[j]) * psi[j])
        iszero(overlap) && throw(ArgumentError("chi and psi are orthogonal: ⟨chi|psi⟩ = 0"))
        chi[j] /= conj(overlap)
    end
    return chi, psi
end