# Parallel MPO on a laptop

This example shows that splitting the Hamiltonian over MPI ranks makes `biorthodmrg`
faster.

## Non-reciprocal XXZ cylinder

Here we look at the XXZ model on a cylinder of $L_x \times L_y$ sites, open along $x$
and periodic along $y$, with non-reciprocal spin flips along $x$,

```math
H = \sum_{\langle ij \rangle} \left[ \frac{1}{2} \left( e^{g_{ij}} S^+_j S^-_i + e^{-g_{ij}} S^-_j S^+_i \right) + \Delta\, S^z_i S^z_j \right],
```

where $g_{ij} = g$ when $j$ is the neighbor of $i$ in the next column and $g_{ij} = 0$
when both sites are in the same column.

Numbering the sites column by column turns every bond along $x$ into a bond of range
$L_y$. Across a cut in the bulk of the chain, $L_y$ sites wait for a partner, each needs
three channels, for $S^+$, $S^-$ and $S^z$, and the MPO adds two identity channels, so
the MPO of $H$ has bond dimension $3L_y + 2$. Rank $p$ of $n$ holds the bonds whose
earlier site lies in a row $y \equiv p + 1 \pmod n$, so each rank's MPO has bond
dimension $3L_y/n + 2$ when $n$ divides $L_y$.

The script `tutorials/mpi_cylinder.jl` runs two sweeps at fixed bond dimension and keeps
the fastest of three runs. Every rank uses one BLAS thread, so $n$ should not exceed the
number of performance cores. Paste this code into your REPL!

```julia
using MPI, GLMakie, BiorthoDMRG

dir, ns = joinpath(pkgdir(BiorthoDMRG), "tutorials", "parallelmpo"), [1, 2, 4]
script = joinpath(dir, "cylinder.jl")
for n in ns
    MPI.mpiexec(cmd -> run(`$cmd -n $n $(Base.julia_cmd()) -t 1 --project=$(Base.active_project()) $script`))
end
t = [parse(Float64, read(joinpath(dir, "time_$n.txt"), String)) for n in ns]

fig = Figure(size = (500, 400))
ax = Axis(fig[1, 1]; xlabel = "ranks", ylabel = "speedup", xticks = ns)
lines!(ax, ns, ns; color = :gray, linestyle = :dot, label = "ideal")
scatterlines!(ax, ns, t[1] ./ t; label = "parallel MPO")
axislegend(ax; position = :lt)

save(joinpath(dir, "speedup.png"), fig)
fig
```

![mpi_speedup](mpi_speedup.png)