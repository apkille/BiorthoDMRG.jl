import BiorthoDMRG.Atoms as DD
import LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

_tag(κ) = replace(string(κ), "." => "p", "-" => "m")

function main(
    output_dir;
    N = 15,
    dovl = 0.2, # spacing in units of λ
    kappas = (1e-3, 1e-1), # trap curvature, units Γ₀³/c²
    mindim = 2,
    maxdims = (64,), # bond dimensions to sweep; agreement between them checks convergence
    nsweeps = 10, # upper bound; a run stops once both residuals fall below residual_tol
    rabis = (0.05, 0.1, 0.15), # coherent drive strength Ω
    detunings = range(-1.0, 3.0; length = 30), # drive detunings Δ to sweep
    kLs = (DD.K0,), # laser wavenumbers to sweep
    cutoff = 1e-20, # BiorthoDMRG truncation cutoff
    eigsolve_maxiter = 50, # GMRES restarts per bond in sssolver
    residual_tol = 1e-6, # in units of Γ₀
    task = nothing,
)

    d  = dovl * DD.LAMBDA
    zc = 0.5 * (N - 1) * d  # chain centre; chain() starts at origin
    pos_vec = DD.chain(N, d; axis = [0.0, 0.0, 1.0])
    z_coords = [r[3] for r in pos_vec]

    bare = DD.Model(
        pos = pos_vec,
        dipole = [1.0, 0.0, 0.0],
    )

    nk_points = 101
    k_vals = range(-pi/d, pi/d; length = nk_points)

    base_name = "steady_trapped_N$(N)_d$(_tag(dovl))"

    # Slice the primary grid (kL, Δ) across cluster nodes
    param_grid = collect(Iterators.product(kLs, detunings))
    if task !== nothing
        param_grid = param_grid[task:task]
    end

    runs = []

    # Each node processes its assigned (kL, Δ) coordinates
    for (kL, Δ) in param_grid
        # Loop through the remaining parameters for this specific slice
        for κ in kappas, mdim in maxdims, rabi in rabis
            model = DD.quadratic(
                DD.driven(bare; rabi, detuning = Δ, kvec = kL .* [0.0, 0.0, 1.0]);
                kappa = κ, axis = [0.0, 0.0, 1.0], center = [0.0, 0.0, zc],
            )

            sol, res = DD.steadystate(
                DD.Biortho(; nsweeps, maxdim = mdim, mindim, cutoff, eigsolve_maxiter,
                             residual_tol, verbose = true),
                model;
                observables = vcat(DD.coherences(N), DD.correlations(N), DD.densities2(N), DD.pairings(N)),
            )

            c = DD.couplings(model)
            Vspan = maximum(LinearAlgebra.diag(c.coh)) - minimum(LinearAlgebra.diag(c.coh))

            # steady state: a single snapshot, sol.t == [Inf]
            fano   = DD.fano(sol, N, 1)
            C_conn = DD.connected(sol, N, 1)
            g2_mat = DD.g2matrix(sol, N, 1)
            C_full = DD.cmatrix(sol, N, 1)  # ⟨σ⁺_i σ⁻_j⟩
            M_mat  = DD.mmatrix(sol, N, 1)  # ⟨σ⁻_i σ⁻_j⟩
            zz_mat = DD.zzmatrix(sol, N, 1) # ⟨σᶻ_i σᶻ_j⟩

            # n(k) = (1/N) * sum_{i,j} e^{+i k (z_i - z_j)} C^{conn}_{ij}; drive at k = +kL
            n_k = zeros(Float64, nk_points)
            for (k_idx, kv) in enumerate(k_vals)
                val = 0.0 + 0.0im
                for i in 1:N, j in 1:N
                    val += exp(im * kv * (z_coords[i] - z_coords[j])) * C_conn[i, j]
                end
                n_k[k_idx] = real(val) / N
            end

            # ⟨nᵢnⱼ⟩ is real for an exact ρ; its imaginary part bounds the error from below
            nn_imag = maximum(abs(imag(sol[Symbol("nn_", i, "_", j)][1])) for i in 1:N for j in i:N)

            key = "kL=$(kL)_det=$(Δ)_kappa=$(κ)_mdim=$(mdim)_rabi=$(rabi)"

            DD.save(joinpath(output_dir, "$(base_name)_$(_tag(key)).h5"),
                    (; k_vals = collect(k_vals), N, dovl,
                       rabi, detuning = Δ, kL, zc,
                       kappa = κ, mdim, Vspan,
                       fano,
                       n_k,
                       C_conn,
                       g2_mat,
                       C_full,
                       M_mat,
                       zz_mat,
                       nn_imag,
                       res,
                       sol);
                    group = key)

            push!(runs, (; kL, Δ, κ, mdim, rabi, Vspan, residuals = res.residuals, nn_imag))
        end
    end

    return (; N, dovl, kLs, detunings, zc, kappas, maxdims, rabis, base_name, runs)
end

output_dir = isempty(ARGS) ? (@__DIR__) : ARGS[1]
task = length(ARGS) < 2 ? nothing : parse(Int, ARGS[2])
mkpath(output_dir)

res = main(output_dir; task)