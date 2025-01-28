include("util.jl")
using MPI
using SparseArrays
using LinearAlgebra

function inverse_matrix_mpi(A::AbstractMatrix, rank, siz, tol::Float64=1e-10, max_iter::Int=2000)
    tr = trace(parallel_matrix_multiply(A', A, rank, siz))
    B = A' / tr

    n, _ = size(A)
    I = Matrix{Float64}(LinearAlgebra.I, n, n)
    i = 1
    while i < max_iter 
        R = I - parallel_matrix_multiply(B, A, rank, siz)

        norm_R = norm(R)
        if norm_R < tol
            return B
        end

        B = parallel_matrix_multiply(I + R, B, rank, siz)
        i = i + 1
    end
    @warn("Przekroczona maksymalna liczba iteracji ($max_iter) z zadaną tolerancją $tol.")
    return B
end

function parallel_matrix_multiply(A, B, rank, siz)
    m, n = size(A)
    nB, p = size(B)

    rows_per_proc = cld(m, siz)

    local_C = Array{Float64}(undef, 1)
    if rows_per_proc*rank+1 <= m
        local_A = A[rank*rows_per_proc+1:min((rank+1)*rows_per_proc, m), :]
        
        local_C = seq_row_multiply(local_A, B)
        local_C = vec(local_C')
        local_C = collect(local_C);
    end

    C = MPI.Gather(local_C, root, comm)

    if rank == root
        C = C[1:n*n]
        C = reshape(C, size(A))'
        C = collect(C)
    else
        C = Matrix{Float64}(undef, n, n)
    end
    
    MPI.Bcast!(C, root, comm)
    return C
end

function seq_row_multiply(A, B)
    n, m = size(A)
    _, p = size(B)

    C = Matrix{Float64}(undef, n, p)

    for i in 1:n
        row_A = view(A, i, :)
        for j in 1:p
            sum_value = 0.0
            for k in 1:m
                sum_value += row_A[k] * B[k, j]
            end
            C[i, j] = sum_value
        end
    end
    return C
end

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
siz = MPI.Comm_size(comm)
root = 0

sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
densities = [0.3, 0.6, 1.0]
tol = 0.01

MPI.Barrier(comm)

results = []
for n in sizes
    for density in densities
        if rank == root
            print("MPI running on $(MPI.Comm_size(comm)) processes.\n")
            print("Test: n = $n, density = $density")
            A = convert(Matrix{Float64}, sprandn(n, n, density))
        else
            A = Matrix{Float64}(undef, n, n)
        end
        MPI.Bcast!(A, root, comm)

        t_elapsed = @elapsed inverse_matrix_mpi(A, rank, siz)

        if rank == root
            push!(results, (size=n, density=density, time=t_elapsed))
            println(", time = $t_elapsed")
        end
    end
end

if rank == root
    save_to_csv(results, "mpi.csv")
end

MPI.Finalize()