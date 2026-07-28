# JuliaQOCO.jl

`JuliaQOCO.jl` is a native Julia primal-dual interior-point solver for convex
quadratic programs with equality, nonnegative-orthant, and second-order-cone
constraints. It provides both a native sparse API and a MathOptInterface/JuMP
optimizer.

The solver uses sparse upper-triangular Hessians, optional Ruiz
equilibration, structured Nesterov-Todd SOC operations, Mehrotra
predictor-corrector steps, and a vendored dependency-free QDLDL backend.

## JuMP usage

Ordinary bridged JuMP use remains supported:

```julia
using JuMP, JuliaQOCO

model = Model(JuliaQOCO.Optimizer)
set_silent(model)
@variable(model, x[1:2] >= 0)
@constraint(model, sum(x) == 1)
@objective(model, Min, sum(x .^ 2))
optimize!(model)
```

For repeated fixed-structure solves, `direct_model` avoids the caching and
bridge layers and is the fastest path:

```julia
model = direct_model(
    JuliaQOCO.Optimizer(;
        verbose = false,
        reuse_solver = true,
        scaling_mode = :once,
        warm_start_mode = :primal_dual,
    ),
)

# Build variables, dynamics, bounds, and cones once.
optimize!(model)

for iteration in 1:max_scp_iterations
    set_normalized_coefficient(dynamics[k], x[j], new_Akj)
    set_normalized_rhs(dynamics[k], new_defect)
    set_upper_bound(trust_radius[k], new_radius)
    set_objective_coefficient(model, trust_radius[k], new_weight)
    optimize!(model)
end
```

JuMP modifications are batched. Individual calls update cached unscaled
numerical storage and allocation-free dirty queues; the complete batch is
committed once at the next `optimize!`. Fixed-pattern matrix entries are
propagated through precomputed maps from the MOI term to native CSC storage,
the cached transpose, KKT storage, and permuted QDLDL numerical storage.
Symbolic analysis and the factor sparsity are retained.

### Generated fixed-pattern KKT backend

For a small, repeatedly solved structure, JuliaQOCO can compile the
fixed-pattern linear algebra into straight-line methods whose sparse
positions are constants:

```julia
model = direct_model(
    JuliaQOCO.Optimizer(;
        verbose = false,
        reuse_solver = true,
        scaling_mode = :once,
        warm_start_mode = :primal_dual,
        kkt_backend = :generated,
    ),
)
```

This remains a normal JuMP workflow: build through JuMP, modify through JuMP
or MOI, and call `optimize!`. The generated method is selected internally
after the first MOI-to-native assembly. Fixed-pattern updates retain the
native solver, symbolic analysis, factor storage, generated code, and
primal-dual iterate.

The generated kernel family includes:

- QDLDL numeric refactorization using the cached symbolic row map;
- row-oriented forward substitution and fixed-pattern triangular solves;
- symmetric Hessian products;
- `A`, `A'`, `G`, and `G'` products used by residual evaluation and
  iterative refinement.

JuliaQOCO's structured O(q) second-order-cone scaling and Nesterov-Todd
matvecs remain in their compact loop form. Benchmarks showed that these
kernels were already too small to justify generating dense cone operations.
This design follows
[QOCOGEN's](https://github.com/qoco-org/qocogen)
[fixed-pattern linear-algebra approach](https://arxiv.org/abs/2503.12658)
while retaining JuliaQOCO's native solver and JuMP/MOI update cache.

`kkt_backend` accepts:

- `:qdldl` (default): cached symbolic QDLDL with its compact numeric loop;
- `:generated`: compile fixed-pattern factorization, solve, and sparse-product
  kernels for this exact problem structure;
- `:auto`: generate only when the KKT dimension and factor nonzeros do not
  exceed `generated_max_dimension` and `generated_max_factor_nnz`.

The limits used by `:auto` default to 384 and 3,000 respectively. First use
of a new pattern can take seconds because Julia must compile a large method;
that cost is included in solver setup time. An identical pattern in the same
Julia process shares the method specialization and does not pay that
compilation cost again. Use `:generated` for long-lived applications that
instantiate or solve the same small structure many times, not for one-off
models or many unrelated large patterns. `:qdldl` remains the best
general-purpose choice.

Inspect the selected backend with:

```julia
MOI.get(
    backend(model),
    MOI.RawOptimizerAttribute("active_kkt_backend"),
)
```

## Directly supported forms

The direct optimizer supports:

- affine and convex quadratic objectives;
- scalar affine `EqualTo`, `LessThan`, `GreaterThan`, and `Interval`;
- variable equality, lower-bound, upper-bound, and interval constraints;
- vector affine and vector-of-variables functions in `Zeros`,
  `Nonnegatives`, and `SecondOrderCone`.

The fixed-structure update path supports:

- `set_objective_coefficient` for existing affine and quadratic terms;
- `set_normalized_coefficient` for existing scalar or vector-affine terms;
- `set_normalized_rhs`;
- changes to existing finite bounds and scalar constraint sets;
- `MOI.ScalarCoefficientChange`,
  `MOI.ScalarQuadraticCoefficientChange`, `MOI.MultirowChange`,
  `MOI.ScalarConstantChange`, and `MOI.VectorConstantChange`;
- `set_start_value` / `MOI.VariablePrimalStart`.

Changing an existing coefficient to zero keeps its reserved sparse entry.
Changing a structurally absent coefficient to nonzero rebuilds the cache.

The following operations deliberately rebuild on the next `optimize!`:

- adding or deleting variables or constraints;
- changing objective function type or objective sense;
- changing cone type or dimension;
- changing between finite and infinite bounds;
- adding a coefficient that has no cached sparse entry;
- changing scaling, KKT regularization, or KKT backend settings.

JuMP variable and constraint references remain valid after a rebuild.

## Scaling and warm starts

`scaling_mode` accepts:

- `:none`: do not equilibrate;
- `:once`: equilibrate during initial assembly and transform later numerical
  updates into the retained scaling;
- `:recompute`: unscale and rerun Ruiz equilibration after matrix changes.

Use `:once` for repeated SCP solves with a fixed pattern. `ruiz_iters`
controls the number of initial Ruiz iterations; its default is zero.

`warm_start_mode` accepts `:none`, `:primal`, `:primal_dual`, and `:adaptive`.
Automatic primal-dual reuse retains the previous internal scaled iterate.
After data changes, the slack is recomputed and cone variables are shifted
strictly into the cone. With unchanged data, the iterate is reused without
perturbation. Explicit JuMP primal starts are also accepted.

## Inspecting reuse

`MOI.RawSolver()` is useful for diagnostics, but is not required for normal
use:

```julia
first_solver = MOI.get(backend(model), MOI.RawSolver())
optimize!(model)
@assert MOI.get(backend(model), MOI.RawSolver()) === first_solver

rebuilds = MOI.get(
    backend(model),
    MOI.RawOptimizerAttribute("rebuild_count"),
)
commit_time = MOI.get(
    backend(model),
    MOI.RawOptimizerAttribute("last_commit_time_sec"),
)
```

Standard JuMP/MOI result access remains available for primal values, dual
values, objective value, statuses, barrier iterations, raw status, and solve
time.

## Native indexed updates

The native API includes full-vector updates and fixed-pattern indexed updates:

```julia
update_P_entries!(solver, indices, values)
update_A_entries!(solver, indices, values)
update_G_entries!(solver, indices, values)
update_c_entries!(solver, indices, values)
update_b_entries!(solver, indices, values)
update_h_entries!(solver, indices, values)
```

Indexed matrix updates require `scaling_mode = :none` or `:once`. These
methods update numerical storage only and defer factorization until the IPM
needs it.

## Example and benchmarks

Run the complete JuMP SCP-style example:

```julia
julia --project=benchmark examples/scp_jump_reuse.jl
```

Run the dedicated JuMP and native benchmarks:

```julia
julia --project=benchmark benchmark/jump_repeated_solves.jl
julia --project=. benchmark/native_repeated_solves.jl
julia --project=benchmark benchmark/scp_like_benchmark.jl
julia --project=benchmark benchmark/generated_kkt_backend.jl
```

The JuMP benchmark separates construction, fresh solves, unchanged cached
solves, vector updates, full and sparse matrix updates, forced rebuilds,
structural rebuilds, and the native indexed-update lower bound. It reports
time, allocations, setup/solve time, iterations, KKT nonzeros, and factor
nonzeros.

The generated-backend benchmark reports first-pattern and already-compiled
same-pattern setup separately, numeric-refactor, triangular-solve,
KKT-product, and complete repeated-solve times, speedups, allocations,
KKT/factor nonzeros, and a JuMP `direct_model` comparison.

## Development

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

The built-in profiling path is enabled with `profile = true` and records
setup, residual, cone scaling, KKT update/factorization, and triangular-solve
timings in `solver.solution.profile`. The non-profiled solve loop uses a
separate timing-free inner kernel.

The bundled QDLDL code is derived from `oxfordcontrol/QDLDL.jl`
(Apache-2.0). See `licenses/QDLDL-LICENSE`.
