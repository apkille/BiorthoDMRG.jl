# ⟨σ⁻_i(t)⟩
cvector(s::Solution, N::Int) =
    ComplexF64[s[Symbol("sm_", i)][k] for k in eachindex(s.t), i in 1:N]

_index(s::Solution) = Dict{Symbol,Int}(n => i for (i, n) in enumerate(s.names))

# uᵀ M v (Bilinear form)
pair(M::AbstractMatrix, u::AbstractVector, v::AbstractVector) = transpose(u) * M * v

# C_{ij} = ⟨σ⁺_i σ⁻_j⟩
function cmatrix(s::Solution, N::Int, k::Int; idx = _index(s))
    C = zeros(ComplexF64, N, N)
    for i in 1:N, j in i:N
        C[i,j] = s.data[k, idx[Symbol("C_", i, "_", j)]]
        C[j,i] = conj(C[i,j])
    end
    return C
end

# ⟨σ⁻_i⟩ at step k
_cvec(s::Solution, N::Int, k::Int, idx) =
    ComplexF64[s.data[k, idx[Symbol("sm_", i)]] for i in 1:N]
    
# ⟨σ⁺_i σ⁻_j⟩ - ⟨σ⁺_i⟩⟨σ⁻_j⟩ (with optional phase e^{i k⋅r_{ij}})
function connected(
    s::Solution, N::Int, k::Int; 
    idx = _index(s), kvec = nothing, pos = nothing)
    haskey(s, Symbol("sm_", 1)) ||
    error("connected: no sm_i in solution; add coherences($N) to observables")
    v = _cvec(s, N, k, idx)
    C = cmatrix(s, N, k; idx) .- conj(v) * transpose(v)
    (kvec === nothing || pos === nothing) && return C
    p = ComplexF64[cis(LinearAlgebra.dot(kvec, r)) for r in pos]
    return p .* C .* transpose(conj(p))
end

# g^{(2)}(r) = 1/N_r ∑_{|i-j|=r} g^{(2)}_{ij}
function g2r(s::Solution, N::Int, k::Int; idx = _index(s), sites = 1:N)
    g = g2matrix(s, N, k; idx)
    rmax = maximum(sites) - minimum(sites)
    tot = zeros(Float64, rmax)
    cnt = zeros(Int, rmax)
    for i in sites, j in sites
        r = j - i
        r > 0 || continue
        tot[r] += g[i,j]
        cnt[r] += 1
    end
    return tot ./ cnt
end

# ⟨σ⁻_i σ⁻_j⟩
function mmatrix(s::Solution, N::Int, k::Int; idx = _index(s))
    M = zeros(ComplexF64, N, N)
    for i in 1:N-1, j in i+1:N
        M[i,j] = M[j,i] = s.data[k, idx[Symbol("M_", i, "_", j)]]
    end
    return M
end

# ⟨σ⁻_i σ⁻_j⟩ - ⟨σ⁻_i⟩⟨σ⁻_j⟩; the diagonal is -⟨σ⁻_i⟩² since σ⁻σ⁻ = 0
function mconnected(s::Solution, N::Int, k::Int; idx = _index(s))
    haskey(s, Symbol("sm_", 1)) ||
    error("mconnected: no sm_i in solution; add coherences($N) to observables")
    v = _cvec(s, N, k, idx)
    return mmatrix(s, N, k; idx) .- v * transpose(v)
end

# ⟨n_i n_j⟩
_nn(s::Solution, N::Int, k::Int, idx) =
    [real(s.data[k, idx[Symbol("nn_", min(i,j), "_", max(i,j))]]) for i in 1:N, j in 1:N]

# C v_m = λ_m v_m (Eigendecomposition for macroscopic occupation)
function mode(C::AbstractMatrix; nmodes::Int = 1, tol::Real = 1e-8)
    F = LinearAlgebra.eigen(LinearAlgebra.Hermitian(Matrix(C)))
    λmin, λmax = minimum(F.values), maximum(F.values)
    p = sortperm(F.values; rev = true)[1:nmodes]
    return F.values[p], F.vectors[:, p]
end

mode(s::Solution, N::Int, k::Int; nmodes::Int = 1, tol::Real = 1e-8, kw...) =
    mode(connected(s, N, k; kw...); nmodes, tol)
    
# g^{(2)}_{ij} = ⟨n_i n_j⟩ / (⟨n_i⟩⟨n_j⟩)
function g2matrix(s::Solution, N::Int, k::Int; idx = _index(s))
    nn = _nn(s, N, k, idx)
    n  = LinearAlgebra.diag(nn)
    g  = nn ./ (n * transpose(n))
    for i in 1:N; g[i,i] = 0.0; end
    return g
end

# ⟨σᶻ_i σᶻ_j⟩ = 4⟨n_i n_j⟩ - 2⟨n_i⟩ - 2⟨n_j⟩ + 1 (connected: 4(⟨n_i n_j⟩ - ⟨n_i⟩⟨n_j⟩))
function zzmatrix(s::Solution, N::Int, k::Int; idx = _index(s), connected::Bool = false)
    nn = _nn(s, N, k, idx)
    n  = LinearAlgebra.diag(nn)
    connected && return 4 .* (nn .- n * transpose(n))
    return 4 .* nn .- 2 .* n .- 2 .* transpose(n) .+ 1
end

# F = (⟨N^2⟩ - ⟨N⟩^2) / ⟨N⟩
function fano(s::Solution, N::Int, k::Int; idx = _index(s))
    nn = _nn(s, N, k, idx)
    Nbar = LinearAlgebra.tr(nn)
    return (sum(nn) - Nbar^2) / Nbar
end