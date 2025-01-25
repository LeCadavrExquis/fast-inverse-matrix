include("util.jl")
using CUDA
using SparseArrays
using LinearAlgebra

function mat_mul_cuda(A::Matrix{Float32}, B::Matrix{Float32})::Matrix{Float32}
    m, n = size(A)
    _, p = size(B)

    a_flat = reduce(vcat, A)
    b_flat = reduce(vcat, B)

    d_a = CuArray{Float32}(undef, m, n)
    copyto!(d_a, A)
    d_b = CuArray{Float32}(undef, n, p)
    copyto!(d_b, B)
    d_c = CuArray{Float32}(undef, m, p)

    threads_per_block = (16, 16)  # 16x16 threads per block
    blocks_per_grid = (ceil(Int, m / threads_per_block[1]), ceil(Int, p / threads_per_block[2]))

    @cuda threads=threads_per_block blocks=blocks_per_grid matmul_kernel(m, n, p, d_a, d_b, d_c)

    C = Array(d_c)
    return C
end

function matmul_kernel(Mdim, Ndim, Pdim, A, B, C)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= Mdim && j <= Pdim
        tmp = 0.0f0
        for k in 1:Ndim
            tmp += A[i + (k - 1) * Mdim] * B[k + (j - 1) * Ndim]
        end
        C[i + (j - 1) * Mdim] = tmp
    end

    return
end

function inverse_matrix_cuda(A::Matrix{Float32}, tol=Float32(1e-4), max_iter=1000)
    n, _ = size(A)
    AT = convert(Matrix{Float32}, A')
    tr = trace(mat_mul_cuda(AT, A))
    B = convert(Matrix{Float32}, AT ./ tr)

    I = Matrix{Float32}(LinearAlgebra.I, n, n)
    i = 1
    while i < max_iter
        R = I - mat_mul_cuda(B, A)

        norm_R = norm(R)
        if norm_R < tol
            return B
        end

        B = mat_mul_cuda(I + R, B)
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
        t_elapsed = @elapsed inverse_matrix_cuda(A, tol)

        push!(results, (size=n, density=density, time=t_elapsed))
        println(", time = $t_elapsed")
    end
end

save_to_csv(results, "cuda.csv")