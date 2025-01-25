using DataFrames
using CSV

function save_to_csv(results, filename="results.csv")
    df = DataFrame(size=Int[], density=Float64[], time=Float64[])
    
    for result in results
        push!(df, (result[:size], result[:density], result[:time]))
    end
    
    CSV.write(filename, df)
    println("Wyniki zapisano do pliku: $filename.")
end

# Macierz musi być kwadratowa, aby obliczyć ślad.
function trace(A::AbstractMatrix{T}) where T
    n, m = size(A)
    tr = zero(T)
    for i in 1:n
        tr += A[i, i]
    end
    return tr
end