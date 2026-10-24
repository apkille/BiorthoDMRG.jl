# BiorthoDMRG.jl

A non-Hermitian operator $H$ has different right and left eigenvectors,

```math
H|\psi\rangle = E|\psi\rangle, \qquad \langle\chi|H = E\langle\chi| .
```

BiorthoDMRG.jl computes an eigenvalue $E$ together with both eigenvectors, as matrix
product states, with a two-site DMRG that updates the two states together. By default
it targets the eigenvalue with the smallest real part.

## Example

```julia
using ITensors, ITensorMPS, BiorthoDMRG
using BiorthoDMRG.Models: xxz

sites = siteinds("S=1/2", 16)
H = xxz(sites; Δ = 0.5, g = 0.1)
psi0 = random_mps(sites; linkdims = 10)
kw = (; nsweeps = 10, maxdim = [20, 40, 80], cutoff = 1e-10)

obs = BiorthoObserver(; H, residual_tol = 1e-6)
E, chi, psi = biorthodmrg(H, psi0; observer = obs, kw...)
```

## Basic explanation

Ordinary DMRG minimizes the Rayleigh quotient
$\langle\psi|H|\psi\rangle / \langle\psi|\psi\rangle$. When $H$ is not normal, the
stationary points of this quotient are not eigenvectors, because such a point would have
to be a left and a right eigenvector at once. Biorthogonal DMRG uses the two-sided
quotient

```math
E = \frac{\langle\chi|H|\psi\rangle}{\langle\chi|\psi\rangle},
```

whose stationary points are exactly the eigenvalues together with their eigenvectors.
Varying $\langle\chi|$ gives $H|\psi\rangle = E|\psi\rangle$, and varying $|\psi\rangle$
gives $\langle\chi|H = E\langle\chi|$.

Both states are MPSs on the same sites, with the same link indices. At bond $\ell$, each
state is a two-site block placed into the rest of its chain,
$|\psi\rangle = P_{\ell\psi}|\phi_\psi\rangle$ and $|\chi\rangle = P_{\ell\chi}|\phi_\chi\rangle$.
Where ordinary DMRG keeps an orthonormal gauge, the pair is kept in the biorthonormal
gauge $P_{\ell\chi}^\dagger P_{\ell\psi} = 1$, so that
$\langle\chi|\psi\rangle = \langle\phi_\chi|\phi_\psi\rangle$. Making $E$ stationary in
the two blocks then gives a local eigenproblem and its adjoint,

```math
P_{\ell\chi}^\dagger H P_{\ell\psi}\,|\phi_\psi\rangle = E\,|\phi_\psi\rangle, \qquad
P_{\ell\psi}^\dagger H^\dagger P_{\ell\chi}\,|\phi_\chi\rangle = \bar E\,|\phi_\chi\rangle .
```

Both are solved with Arnoldi, and the blocks are normalized so that
$\langle\phi_\chi|\phi_\psi\rangle = 1$. This is possible whenever $E$ is a simple
eigenvalue of the local operator. The blocks are then split back into two sites in a way
that keeps the gauge, and the sweep moves on, left to right and back, as in two-site
DMRG.

It is important to say that there is no variational principle in biorthogonal DMRG. $E$ is a ratio of linear functions of the components
of $\langle\chi|$ and of $|\psi\rangle$, so it is holomorphic in each, and its real part
has no local minimum. Every stationary point is a saddle. $E$ is therefore not an upper
bound and need not decrease from sweep to sweep.

The accuracy depends on how far $H$ is from normal, through the condition number

```math
\kappa = \frac{\|\chi\|\,\|\psi\|}{|\langle\chi|\psi\rangle|} \geq 1 .
```

Of course, it equals 1 when $|\chi\rangle \propto |\psi\rangle$, and it diverges at an exceptional
point. There the left and right eigenvectors become orthogonal and the normalization
above fails.

The error in $E$ is bilinear in the errors of the two states. For any right and left
eigenvectors $|R_j\rangle$ and $\langle L_j|$ of an exact eigenvalue $E_j$,

```math
E - E_j = \frac{\langle\chi - L_j|(H - E_j)|\psi - R_j\rangle}{\langle\chi|\psi\rangle},
\qquad
|E - E_j| \le \|H - E_j\|\,\kappa\,\frac{\|\chi - L_j\|}{\|\chi\|}\,\frac{\|\psi - R_j\|}{\|\psi\|} .
```

So $E$ converges as the product of the two state errors, amplified by $\kappa$. An accurate
$E$ alone therefore does not certify accurate states, and convergence should be judged on the
residuals $\|H\psi - E\psi\|/\|\psi\|$ and $\|H^\dagger\chi - \bar E\chi\|/\|\chi\|$.