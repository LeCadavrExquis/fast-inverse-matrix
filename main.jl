include("util.jl")
using SparseArrays
using LinearAlgebra

function inverse_matrix(A::AbstractMatrix, tol::Float64=1e-10, max_iter::Int=1000)
    B = A' / tr(A' * A)

    i = 1
    while i < max_iter
        R = I - B * A

        norm_R = norm(R)
        if norm_R < tol
            return B
        end

        B = (I + R) * B
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

        t_elapsed = @elapsed inverse_matrix(A, tol)

        push!(results, (size=n, density=density, time=t_elapsed))
        println(", time = $t_elapsed")
    end
end

save_to_csv(results, "pure_julia.csv")