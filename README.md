# JuliaQOCO.jl

`JuliaQOCO.jl` is a native Julia implementation of the QOCO primal-dual interior-point method for convex quadratic programs with equality, nonnegative orthant, and second-order cone constraints.

The package exposes:

- a native solver API for QOCO-standard-form problems
- a `MathOptInterface` / JuMP optimizer

The implementation mirrors the upstream `qoco` solver structure closely:

- sparse upper-triangular quadratic Hessian storage
- Ruiz equilibration
- Nesterov-Todd scaling for SOCs
- Mehrotra predictor-corrector updates
- sparse KKT solves through `QDLDL.jl` with in-place numeric refactorization

## Scope

The current target problem class is

```math
\begin{aligned}
\min_x \quad & \tfrac{1}{2} x^\top P x + c^\top x \\
\text{s.t.} \quad & A x = b \\
& G x \preceq_{\mathcal C} h
\end{aligned}
```

with

```math
\mathcal C = \mathbb{R}_+^l \times \mathcal{Q}^{q_1} \times \cdots \times \mathcal{Q}^{q_N}.
```

## Development

Run the package tests with:

```julia
julia --project=. -e "using Pkg; Pkg.test()"
```
