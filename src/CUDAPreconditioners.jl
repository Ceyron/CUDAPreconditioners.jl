"""
Wrappers for incomplete factorizations from the CUSPARSE library together with
convenience functions to use them as valid preconditioners in most Julia
iterative solvers.

Based on the tutorial from Krlov.jl
https://juliasmoothoptimizers.github.io/Krylov.jl/stable/gpu/#Example-with-a-symmetric-positive-definite-system
"""

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra


struct CuILU0CSC{
    Tv<:Number,
    Ti<:Integer
}
    factorization::CuSparseMatrixCSC{Tv, Ti}

    CuILU0CSC(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv<:Number, Ti<:Integer} = new{Tv, Ti}(ilu02(A, 'O'))
end

function LinearAlgebra.ldiv!(y, P::CuILU0CSC, x)
    copyto!(y, x)
    sv2!('N', 'L', 'N', 1.0, P.factorization, y, 'O')
    sv2!('N', 'U', 'U', 1.0, P.factorization, y, 'O')
    return y
end

struct CuILU0CSR{
    Tv<:Number,
    Ti<:Integer
}
    factorization::CuSparseMatrixCSR{Tv, Ti}

    CuILU0CSR(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv<:Number, Ti<:Integer} = new{Tv, Ti}(ilu02(A, 'O'))
end

function LinearAlgebra.ldiv!(y, P::CuILU0CSR, x)
    copyto!(y, x)
    sv2!('N', 'L', 'U', 1.0, P.factorization, y, 'O')
    sv2!('N', 'U', 'N', 1.0, P.factorization, y, 'O')
    return y
end