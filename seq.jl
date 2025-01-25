include("util.jl")
using SparseArrays
using LinearAlgebra

function mat_mul_seq(A::Array{T}, B::Array{T}) where T
    m, n = size(A)
    _, p = size(B)
    C = Matrix{Float32}(undef, p, m)
    for i in 1:n
        for j in 1:m
            tmp = zero(T)
            for k in 1:p
                tmp += A[(i-1) * n + k] * B[(k-1) * p + j]
            end
            C[(i-1) * n + j] = tmp
        end
    end
    return C
end

function inverse_matrix_seq(A::Matrix{Float32}, tol=Float32(1e-4), max_iter=1000)
    n, _ = size(A)
    AT = convert(Matrix{Float32}, A')
    tr = trace(mat_mul_seq(AT, A))
    B = convert(Matrix{Float32}, AT ./ tr)

    I = Matrix{Float32}(LinearAlgebra.I, n, n)
    i = 1
    while i < max_iter 
        R = I - mat_mul_seq(B, A)

        norm_R = norm(R)
        if norm_R < tol
            return B
        end

        B = mat_mul_seq(I + R, B)
        i = i + 1
    end

    @warn("Przekroczona maksymalna liczba iteracji ($max_iter) z zadaną tolerancją $tol.")
    return B
end

sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
densities = [0.3, 0.6, 1.0]
tol = 0.01

results = []
for n in sizes
    for density in densities
        print("Test: n = $n, density = $density")
        A = convert(Matrix{Float32}, sprandn(n, n, density))

        t_elapsed = @elapsed inverse_matrix_seq(A, tol)

        push!(results, (size=n, density=density, time=t_elapsed))
        println(", time = $t_elapsed")
    end
end

save_to_csv(results, "seq.csv")