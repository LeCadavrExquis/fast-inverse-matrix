# Matrix Inversion and Multiplication using Julia

This project demonstrates different implementations of matrix multiplication and inversion using three distinct approaches:
- **OpenCL**
- **Julia**
- **CUDA**

## Requirements

- **Julia** (v1.0 or higher)
- **OpenCL** library for Julia
- **SparseArrays** and **LinearAlgebra** standard Julia packages
- **DataFrames** and **CSV** for result storage

## Files Overview

### `openCL.jl`

This file also contains kernel source code for the matrix multiplication (`mmul`), written in OpenCL C.

### `main.jl`

This file contains a standard Julia implementation for matrix inversion and matrix multiplication.

### `seq.jl`

### `util.sl`
Functions:
- `save_to_csv`
- `trace`

## How to Run

1. **Install required packages**:
   ```julia
   using Pkg
   Pkg.add("OpenCL")
   Pkg.add("SparseArrays")
   Pkg.add("LinearAlgebra")
   Pkg.add("CSV")
   Pkg.add("DataFrames")
