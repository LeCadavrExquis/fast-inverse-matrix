include("util.jl")
using Distributed
addprocs(3)
@everywhere using SharedArrays
using LinearAlgebra
using SparseArrays

function inverse_matrix_distributed(A::AbstractMatrix, tol::Float64=1e-10, max_iter::Int=1000)
    tr = trace(parallel_row_multiply(A', A))
    B = A' / tr

    i = 1
    while i <= max_iter
        R = I - parallel_row_multiply(B, A)

        norm_R = norm(R)
        if norm_R < tol
            return B
        end

        B = parallel_row_multiply((I + R), B)
        i += 1
    end
    @warn("Przekroczona maksymalna liczba iteracji ($max_iter) z zadaną tolerancją $tol.")
    return B
end

function parallel_row_multiply(A, B)
    n, m = size(A)
    _, p = size(B)

    C = SharedArray{Float64}(n, p)

    @sync @distributed for i in 1:n
        row_A = A[i, :]
        for j in 1:p
            C[i, j] = sum(row_A[k] * B[k, j] for k in 1:m)
        end
    end

    return C
end

println("processes = $(nprocs())")

sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
densities = [0.3, 0.6, 1.0]
tol = 0.01

results = []
for n in sizes
    for density in densities
        print("Test: n = $n, density = $density")
        A = convert(Matrix{Float32}, sprandn(n, n, density))

        t_elapsed = @elapsed inverse_matrix_distributed(A, tol)

        push!(results, (size=n, density=density, time=t_elapsed))
        println(", time = $t_elapsed")
    end
end

save_to_csv(results, "distributed.csv")