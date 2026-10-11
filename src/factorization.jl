"""
    _biortho_basis(phi_psi, phi_chi, is; biortho_cutoff, tags, svd_kwargs...)

Biorthonormal bases for `phi_psi` and `phi_chi` on the indices `is`.

Each block is truncated by an SVD with `svd_kwargs`. The two kept bases are then
rotated and rescaled, using the SVD of their overlap, so that `Xchi† Xpsi` is the
identity on the new link index, which carries `tags`. 

The singular values of the overlap are the cosines of the angles between the 
kept subspaces; those at or below `biortho_cutoff` are dropped. 
Also returns both truncation spectra and `sigma_min`, the smallest kept cosine. 
Dense indices only.
"""
function _biortho_basis(
    phi_psi::ITensor, 
    phi_chi::ITensor, 
    is; 
    biortho_cutoff::Real, 
    tags, 
    svd_kwargs...
)
    (hasqns(phi_psi) || hasqns(phi_chi)) &&
        throw(
            ArgumentError("biorthogonal factorization is implemented for dense indices only")
        )
    Fpsi = ITensors.svd(phi_psi, is; svd_kwargs...)
    Fchi = ITensors.svd(phi_chi, is; svd_kwargs...)
    upsi, uchi = commonind(Fpsi.U, Fpsi.S), commonind(Fchi.U, Fchi.S)
    m = prod(dim, is)
    Upsi = reshape(array(Fpsi.U, is..., upsi), m, dim(upsi))
    Uchi = reshape(array(Fchi.U, is..., uchi), m, dim(uchi))
    F = LinearAlgebra.svd!(Uchi' * Upsi)
    r = count(>(biortho_cutoff), F.S)
    r == 0 && error("kept left and right subspaces are orthogonal at this bond")
    w = Diagonal(inv.(sqrt.(view(F.S, 1:r))))
    Xpsi = rmul!(Upsi * view(F.Vt, 1:r, :)', w)
    Xchi = rmul!(Uchi * view(F.U, :, 1:r), w)
    l = Index(r, tags)
    dims = (dim.(is)..., r)
    return itensor(reshape(Xpsi, dims), is..., l),
        itensor(reshape(Xchi, dims), is..., l),
        (; spec_psi = Fpsi.spec, spec_chi = Fchi.spec, sigma_min = F.S[r])
end

# One orthonormal basis for both blocks: the left singular vectors of the two
# normalized blocks stacked along an auxiliary index. Both states receive the same
# isometry, so the local operator is an orthogonal compression of `H`.
function _shared_basis(phi_psi::ITensor, phi_chi::ITensor, is; tags, svd_kwargs...)
    c = Index(2, "stack")
    stacked = phi_psi / norm(phi_psi) * onehot(c => 1) + phi_chi / norm(phi_chi) * onehot(c => 2)
    F = ITensors.svd(stacked, is; svd_kwargs...)
    u = commonind(F.U, F.S)
    U = replaceind(F.U, u, Index(dim(u), tags))
    return U, U, (; spec_psi = F.spec, spec_chi = F.spec, sigma_min = 1.0)
end

"""
    biorthoreplacebond!(chi, psi, ℓ, phi_chi, phi_psi; ortho, basis=:biorthonormal, biortho_cutoff=0, svd_kwargs...)

Split the two-site blocks `phi_chi` and `phi_psi` back into sites `ℓ` and `ℓ+1`
of `chi` and `psi`, truncating with `svd_kwargs`. With `ortho="left"`, site `ℓ`
receives the new bases and site `ℓ+1` the remainders, so the window moves right;
`ortho="right"` does the reverse. Both states get the same new link index.

`basis=:shared` gives both states one orthonormal basis that captures both
blocks. The local operator is then an orthogonal compression of `H`, whose
eigenvalues stay within the numerical range of `H`. `basis=:biorthonormal` gives
each state its own basis, which is more compact, but the local operator then has
eigenvalues that can lie far outside the spectrum of `H`.

Returns the spectra of both blocks and `sigma_min`, which is `1` for `:shared`.
For `:biorthonormal`, if no cosine is dropped and both blocks keep the same
dimension, the relative error of each block is at most
`(1 + 1/sigma_min) * sqrt(truncerr)`.
"""
function biorthoreplacebond!(
        chi::MPS, psi::MPS, ℓ::Int, phi_chi::ITensor, phi_psi::ITensor;
        ortho::String, basis::Symbol = :biorthonormal, biortho_cutoff::Real = 0, svd_kwargs...,
    )
    site, other = ortho == "left" ? (ℓ, ℓ + 1) :
        ortho == "right" ? (ℓ + 1, ℓ) :
        throw(ArgumentError("ortho must be \"left\" or \"right\", got \"$ortho\""))
    is = uniqueinds(psi[site], psi[other])
    tags = "Link,l=$ℓ"
    Xpsi, Xchi, info = if basis === :shared
        _shared_basis(phi_psi, phi_chi, is; tags, svd_kwargs...)
    elseif basis === :biorthonormal
        _biortho_basis(phi_psi, phi_chi, is; biortho_cutoff, tags, svd_kwargs...)
    else
        throw(ArgumentError("basis must be :shared or :biorthonormal, got :$basis"))
    end
    psi[site], chi[site] = Xpsi, Xchi
    psi[other], chi[other] = dag(Xchi) * phi_psi, dag(Xpsi) * phi_chi
    return info
end