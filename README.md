# BiorthoDMRG.jl

A non-Hermitian $H$ has distinct right and left eigenvectors,

```math
H|\psi\rangle = E|\psi\rangle, \qquad \langle\chi|H = E\langle\chi| .
```

This package finds the ground state energy of $H$ using DMRG.

## Usage

```julia
using ITensors, ITensorMPS, BiorthogonalDMRG

obs = BiorthoObserver(; H, residual_tol = 1e-6)
E, chi, psi = biorthodmrg(
    H, 
    psi0; 
    nsweeps = 10, 
    maxdim = [20, 40, 80], 
    cutoff = 1e-10, 
    observer = obs
)

E1, chi1, psi1 = biorthodmrg(
    H, 
    [(chi, psi)], 
    psi0, 
    psi0; 
    weight = 10.0,
    nsweeps = 10, 
    maxdim = [20, 40, 80], 
    cutoff = 1e-10
)
```

## Implementation details

`biorthodmrg` represents both as MPSs, `psi` and `chi`, on the same sites and with the same link indices.

The local problem to solve, is that bond $\ell$, each state is written as a map applied to a two-site block:

```math
|\psi\rangle = P_{\ell\psi}|\phi_\psi\rangle, \qquad |\chi\rangle = P_{\ell\chi}|\phi_\chi\rangle .
```

$P_{\ell\psi}$ consists of every tensor of `psi` except those on sites $\ell$ and $\ell+1$, and $P_{\ell\chi}$ likewise for `chi`.

The pair is kept in the biorthonormal gauge $P_{\ell\chi}^\dagger P_{\ell\psi} = 1$. In this gauge, contracting `dag(chi[k])` with `psi[k]` gives the identity on the remaining link: over the left link and site index for $k < \ell$, and over the site index and right link for $k > \ell + 1$.

The energy is the two-sided Rayleigh quotient $E = \langle\chi|H|\psi\rangle / \langle\chi|\psi\rangle$. Making it stationary with respect to $\langle\phi_\chi|$ and $|\phi_\psi\rangle$, and using the gauge, gives two eigenproblems for one operator:

```math
P_{\ell\chi}^\dagger H P_{\ell\psi}|\phi_\psi\rangle = E|\phi_\psi\rangle, \qquad
P_{\ell\psi}^\dagger H^\dagger P_{\ell\chi}|\phi_\chi\rangle = \bar{E}|\phi_\chi\rangle .
```

`BiorthoProjMPO` contracts $P_{\ell\chi}^\dagger H P_{\ell\psi}$ from mixed environments, with `chi` on the bra side and `psi` on the ket side. Its adjoint uses the same environments, read from the other side.

`biorthoproblemsolver` solves the right problem with Arnoldi and then the left
problem with Arnoldi targeted at the conjugate eigenvalue, and normalizes so that

```math
\langle\phi_\chi|\phi_\psi\rangle = \langle\chi|\psi\rangle = 1 .
```

This normalization is possible whenever $E$ is a simple eigenvalue.  Of course, it fails at exceptional points, where left and right eigenvectors become orthogonal.

`biorthoreplacebond!` splits $|\phi_\psi\rangle$ and $|\phi_\chi\rangle$ back into two sites in three steps:

1. Each block is truncated by an SVD.
2. The two kept bases are made biorthonormal using the SVD of their overlap. The singular values of that overlap are the cosines of the angles between the kept left and right subspaces.
3. The resulting split is an oblique projection, not an orthogonal one, so truncation errors are amplified. If no cosine is dropped and both kept dimensions are equal, the relative error of each block is at most $(1 + 1/\sigma_{\min})\sqrt{\mathrm{truncerr}}$, where $\sigma_{\min}$ is the smallest kept cosine. `biortho_cutoff` drops small cosines to keep the projection well conditioned.


The sweep alternates local solves and factorizations, left to right and back, as in two-site DMRG. The environments of both states are cached, and each step recomputes only those touching the two sites it changed.

Sadly, for biorthogonality, we cannot use the variational principle. The two-sided quotient has saddle points but no minimum, so $E$ is not monotone across sweeps and is not a bound. That said, we can use two principles that take the place of the variational principle.

1. Fixed points: a converged pair satisfies $P_{\ell\chi}^\dagger (H - E)|\psi\rangle = 0$ at every bond.
2. Error bound: let $(E_j, \langle L_j|, |R_j\rangle)$ be the exact eigentriple. The error in $E$ is bilinear in the errors of the two states:

```math
|E - E_j| \le \|H - E_j\| \, \kappa \, \frac{\|\chi - L_j\|}{\|\chi\|} \, \frac{\|\psi - R_j\|}{\|\psi\|},
\qquad \kappa = \frac{\|\chi\| \, \|\psi\|}{|\langle\chi|\psi\rangle|} \ge 1 .
```

Convergence is therefore judged on the residuals $\|H\psi - E\psi\|$ and $\|H^\dagger\chi - \bar{E}\chi\|$. `BiorthoObserver` records them together with $\kappa$ and stops the run once both fall below `residual_tol`.