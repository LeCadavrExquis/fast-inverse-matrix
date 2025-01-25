include("util.jl")

using OpenCL
using SparseArrays
using LinearAlgebra

function mat_mul_openCl(A::Matrix{Float32}, B::Matrix{Float32}):: Matrix{Float32}
    m, n = size(A)
    _, p = size(B)

    a_flat = reshape(A, (m * n,))
    b_flat = reshape(B, (n * p,))

    d_a = CLArray(a_flat; access=:r)
    d_b = CLArray(b_flat; access=:r)
    d_c = CLArray{Float32}(undef, m * p; access=:w)

    global_size = (m, p)
    local_size = (min(m, 16), min(p, 16))

    C = zeros(Float32, p * m) 

    cl.queue!(:profile) do
        evt = clcall(mmul, Tuple{Int32, Int32, Int32, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}},
                    m, n, p, d_a, d_b, d_c; global_size, local_size)
        wait(evt)
        cl.copy!(C, d_c)
    end

    return reshape(C, p, m)
end

function inverse_matrix_openCl(A::Matrix{Float32}, tol=Float32(1e-4), max_iter=1000)
    n, _ = size(A)
    AT = convert(Matrix{Float32}, A')
    tr = trace(mat_mul_openCl(AT, A))
    B = convert(Matrix{Float32}, AT ./ tr)

    I = Matrix{Float32}(LinearAlgebra.I, n, n)
    i = 1
    while i < max_iter 
        R = I - mat_mul_openCl(B, A)

        norm_R = norm(R)
        if norm_R < tol
            return B
        end

        B = mat_mul_openCl(I + R, B)
        i = i + 1
    end

    @warn("Przekroczona maksymalna liczba iteracji ($max_iter) z zadaną tolerancją $tol.")
    return B
end

# Inicializacja programu
kernel_source = """
__kernel void mmul(
    const int Mdim,
    const int Ndim, 
    const int Pdim,
    __global float* A,
    __global float* B,
    __global float* C)
{
    int k;
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp;
    if ((i < Ndim) && (j < Mdim))
    {
        tmp = 0.0f;
        for (k = 0; k < Pdim; k++)
            tmp += A[i*Ndim+k] * B[k*Pdim+j];
        C[i*Ndim+j] = tmp;
    }
}
"""
prg  = cl.Program(source=kernel_source) |> cl.build!
mmul = cl.Kernel(prg, "mmul")

sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
densities = [0.3, 0.6, 1.0]
tol = 0.01

results = []
for n in sizes
    for density in densities
        print("Test: n = $n, density = $density")
        A = convert(Matrix{Float32}, sprandn(n, n, density))

        t_elapsed = @elapsed inverse_matrix_openCl(A, tol)

        push!(results, (size=n, density=density, time=t_elapsed))
        println(", time = $t_elapsed")
    end
end

save_to_csv(results, "opencl.csv")