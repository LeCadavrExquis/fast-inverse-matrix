# Fast iterative matrix inversion in Julia

The project shows different implementations of parallelization of computations using distinct approaches:
- **Julia**
- **OpenCL**
- **CUDA**
- **MPI**

![Alt text](doc/image.png)

## Required packages:
   ```julia
   using Pkg
   Pkg.add("SparseArrays")
   Pkg.add("LinearAlgebra")
   Pkg.add("CSV")
   Pkg.add("DataFrames")
   Pkg.add("OpenCL")
   Pkg.add("CUDA")
   Pkg.add("MPI")
   Pkg.add("Distributed")
   Pkg.add("SharedArrays")
