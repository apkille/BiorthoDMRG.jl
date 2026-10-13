allground(N::Int)  = fill(:g, N)
allexcited(N::Int) = fill(:e, N)
single_excitation(N::Int, idx::Int) = [i == idx ? :e : :g for i in 1:N]