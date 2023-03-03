# CUDAPreconditioners.jl
Convenience wrappers to incomplete factorizations from CUSPARSE to be used for
iterative solvers of sparse linear systems on the GPU. The implementations are
adapted from the [great
tutorial](https://juliasmoothoptimizers.github.io/Krylov.jl/stable/gpu/#Example-with-a-symmetric-positive-definite-system)
of the Krylov.jl package.

Supports both **Incomplete Cholesky factorization with zero fill-in** (`ic0`)
for symmetric positive definite matrices and **Incomplete LU factorization with zero fill-in** (`ilu0`).

The created preconditioners can be used with any iterative linear solver that supports right-preconditioning (e.g. `gmres` from Krylov.jl).

## Examples

### Solving the 1D Heat Equation on the GPU

The 1D Heat Equation

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

with homogeneous Dirichlet boundary conditions

$$
u(0,t) = u(1,t) = 0
$$

and a backward-in-time central-in-space discretization (implicit Euler) requires the solution to a linear system of equations in each iteration

$$
A u^{[t+1]} = u^{[t]}
$$

for the vector of (interior) degrees of freedom. The matrix $A$ is tridiagonal and sparse. It has the structure

$$
A = \begin{pmatrix}
1 + 2 \alpha \frac{\Delta t}{\Delta x^2} & -\alpha \frac{\Delta t}{\Delta x^2} & 0 & \dots & 0 \\
-\alpha \frac{\Delta t}{\Delta x^2} & 1 + 2 \alpha \frac{\Delta t}{\Delta x^2} & -\alpha \frac{\Delta t}{\Delta x^2} & \dots & 0 \\
0 & -\alpha \frac{\Delta t}{\Delta x^2} & 1 + 2 \alpha \frac{\Delta t}{\Delta x^2} & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & 1 + 2 \alpha \frac{\Delta t}{\Delta x^2}
\end{pmatrix}
$$

It is unconditionally stable for all $\alpha$ and $\Delta t$.

Let's first assemble the matrix $A$ on the CPU and set up an initial condition.

```julia
using SparseArrays
using LinearAlgebra

N = 10_000  # Interior DoF without the 2 boundary points
Δx = 1 / (N + 1)
Δt = 0.1
α = 1.0

A = spdiagm(-1 => -α * Δt / Δx^2 * ones(N-1),
             0 => 1 .+ 2 * α * Δt / Δx^2 * ones(N),
             1 => -α * Δt / Δx^2 * ones(N-1))

mesh = range(0.0, 1.0, length=N+2)
u = ifelse.((mesh .> 0.2) .& (mesh .< 0.4), 1.0, 0.0)[2:end-1]
```

Let's move the matrix $A$ to the GPU and create a preconditioner for it.

```julia
using CUDA
using CUDA.CUSPARSE
using CUDAPreconditioners

A_gpu = CuSparseMatrixCSR(A)
P_r_gpu = ic0(A_gpu)
```

Then, we can solve the linear system on the GPU using the `cg` (conjugate
gradient) solver from
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

```julia
using Krylov

u_gpu = CuArray(u)

# Solve without a preconditioner
u_gpu_next_no_preconditioner, solve_stats_no_preconditioner = cg(
    A_gpu,
    u_gpu,
)

# Solve with the preconditioner
u_gpu_next_with_preconditioner, solve_stats_with_preconditioner = cg(
    A_gpu,
    u_gpu,
    M=P_r_gpu,
    ldiv=true,
)

@show solve_stats_no_preconditioner.niter  # -> 9999
@show solve_stats_with_preconditioner.niter  # -> 7
```

The preconditioned version takes way fewer iterations to converge.

