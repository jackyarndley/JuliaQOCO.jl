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
- sparse KKT solves through a vendored `QDLDL.jl` backend with in-place numeric refactorization

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

## Verbose output

The native solver and the `MathOptInterface` optimizer now print the upstream-style QOCO convergence table by default.

- Native API: disable it with `Settings(verbose = false)`
- Native API: redirect it with `Settings(output = io)` while keeping terminal output as the default via `stdout`
- MOI / JuMP: disable it with `MOI.set(model, MOI.Silent(), true)`

The bundled linear-system code is derived from `oxfordcontrol/QDLDL.jl` (Apache-2.0). See `licenses/QDLDL-LICENSE` for the vendored license text.

## Development

Run the package tests with:

```julia
julia --project=. -e "using Pkg; Pkg.test()"
```

For solver work, keep the inner loop on the native path and defer the full MOI suite until a checkpoint:

```julia
julia --project=. -e "using Pkg; Pkg.test(; test_args=[\"native\"])"
```

The solver also now supports lightweight built-in timing breakdowns through `Settings(profile = true)`, which records setup, KKT, scaling, stopping, predictor-corrector, and linear-solve timings on `solver.solution.profile`.

For repeated-solve benchmarking without the MOI wrapper, run:

```julia
julia --project=. benchmark/native_repeated_solves.jl
```

For a more SCP-like repeated benchmark with stage-coupled dynamics updates, SOC constraints, and `BenchmarkTools` measurements, run:

```julia
julia --project=benchmark benchmark/scp_like_benchmark.jl
```
