"""
Wrappers for incomplete factorizations from the CUSPARSE library together with
convenience functions to use them as valid preconditioners in most Julia
iterative solvers.

Based on the tutorial from Krlov.jl
https://juliasmoothoptimizers.github.io/Krylov.jl/stable/gpu/#Example-with-a-symmetric-positive-definite-system
"""
module CUDAPreconditioners

using CUDA
using CUDA.CUSPARSE

import LinearAlgebra.ldiv!

export ilu0, ldiv!


struct CuILU0CSC{
    Tv<:Number,
    Ti<:Integer
}
    factorization::CuSparseMatrixCSC{Tv, Ti}

    CuILU0CSC(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv<:Number, Ti<:Integer} = new{Tv, Ti}(ilu02(A, 'O'))
end

function ldiv!(y, P::CuILU0CSC, x)
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

function ldiv!(y, P::CuILU0CSR, x)
    copyto!(y, x)
    sv2!('N', 'L', 'U', 1.0, P.factorization, y, 'O')
    sv2!('N', 'U', 'N', 1.0, P.factorization, y, 'O')
    return y
end

"""
    ilu0(A:<{CuSparseMatrixCSC, CuSparseMatrixCSR})

Compute the incomplete LU factorization of `A` using the CUSPARSE library,
returning a struct for which `ldiv!` can be used as a preconditioner.
"""
function ilu0(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv<:Number, Ti<:Integer}
    return CuILU0CSC(A)
end

function ilu0(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv<:Number, Ti<:Integer}
    return CuILU0CSR(A)
end

end # module CUDAPreconditioners
