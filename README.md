# JuliaQOCO.jl

JuliaQOCO is a pure-Julia MathOptInterface optimizer for convex quadratic
programs with equality, nonnegative-orthant, and second-order-cone
constraints. The supported public interface is `JuliaQOCO.Optimizer`.

```julia
using JuMP, JuliaQOCO

model = Model(JuliaQOCO.Optimizer)
set_silent(model)
@variable(model, x[1:2] >= 0)
@constraint(model, sum(x) == 1)
@objective(model, Min, sum(x[i]^2 for i in 1:2))
optimize!(model)
```

For sequential convex programming, build a direct model once and update the
existing functions:

```julia
model = direct_model(JuliaQOCO.Optimizer(
    verbose = false,
    scaling_mode = :once,
    warm_start_mode = :primal_dual,
))
# Add dynamics, bounds, trust regions, virtual controls, and SOCs once.
optimize!(model)
set_normalized_rhs(dynamics[k], new_defect)
set_normalized_coefficient(dynamics[k], x[j], new_jacobian)
set_objective_coefficient(model, x[j], new_weight)
optimize!(model)
```

Complete scalar and vector affine function replacements are numerical updates
when their function type, dimension, support, and cone/bound pattern are
unchanged. Values are committed through precomputed CSC, transpose, KKT, and
QDLDL maps. The solver object, AMD ordering, symbolic factorization, factor
storage, frozen scaling, and warm start are retained. A zero value remains a
reserved structural entry; a genuinely new nonzero support entry causes one
rebuild on the next solve.

Use `scaling_mode = :once` for repeated fixed-pattern solves. `:none` disables
equilibration and `:recompute` reruns equilibration after matrix updates.
`warm_start_mode` supports `:none`, `:primal`, `:primal_dual`, and
`:adaptive`. The solver restores its best finite iterate when an iteration
limit, numerical failure, or severe stall degrades the current iterate.

The direct optimizer supports feasibility, affine, and convex quadratic
objectives; minimization; valid convex maximization conversion; scalar affine
and variable bounds/equalities; and vector zero, nonnegative, and SOC
constraints. Indefinite quadratic objectives, unsupported cones, invalid
dimensions, non-finite data, and duplicate terms are rejected.

QOCO does not compute homogeneous-embedding infeasibility certificates.
`MOI.ITERATION_LIMIT`, `MOI.NUMERICAL_ERROR`, and inaccurate solutions must
not be interpreted as certified infeasibility. Dual objective values are not
claimed by this optimizer until a correct implementation is available.

Useful diagnostics are available through `MOI.RawOptimizerAttribute`,
including `rebuild_count`, `structure_generation`, `last_commit_time_sec`,
`commit_count`, `symbolic_rebuild_count`, `last_rebuild_reason`,
`regularized_entries`, `dynamic_regularizations`, and the cached solver's
`Solution` profile. `MOI.RawSolver()` may be used to verify identity and factor
reuse in integration tests.

The benchmark hierarchy lives under `benchmark/`. The representative
JuMP/SCP-style workflow is:

```text
julia --project=benchmark examples/scp_jump_reuse.jl
julia --project=benchmark benchmark/jump_repeated_solves.jl
```

JuliaQOCO is based on the pure-Julia numerical method described in the QOCO
paper. The vendored sparse direct factorization is derived from
`oxfordcontrol/QDLDL.jl` under Apache-2.0; see `licenses/QDLDL-LICENSE`.
