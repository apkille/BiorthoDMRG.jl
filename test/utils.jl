using ITensors, ITensorMPS

# Dense matrix of an MPO and dense vector of an MPS on the sites `s`
dense_matrix(A::MPO, s) = reshape(Array(prod(A[k] for k in eachindex(s)), prime.(s)..., s...), prod(dim, s), prod(dim, s))
dense_vector(x::MPS, s) = vec(Array(prod(x[k] for k in eachindex(s)), s...))

# Norm of an MPS without the warning `norm` gives for complex states
nrm(x::MPS) = sqrt(real(inner(x, x)))