# JuliaQOCO.jl

JuliaQOCO is a pure-Julia MathOptInterface optimizer for convex quadratic
programs with equality, nonnegative-orthant, and second-order-cone
constraints. It is built for sequential convex programming: one model is
constructed once and then re-solved many times with changed numerical data on
an unchanged sparsity pattern.

The supported public interface is `JuliaQOCO.Optimizer`. Everything else,
including `CoreSolver`, is internal and may change without notice.

```julia
using JuMP, JuliaQOCO

model = Model(JuliaQOCO.Optimizer)
set_silent(model)
@variable(model, x[1:2] >= 0)
@constraint(model, sum(x) == 1)
@objective(model, Min, sum(x[i]^2 for i in 1:2))
optimize!(model)
```

## Supported problem form

```text
minimize    (1/2) x' P x + c' x
subject to  A x = b
            h - G x in K
```

`K` is the nonnegative orthant of dimension `l` followed by second-order cones
of dimensions `q[1], q[2], ...`, in that order. `P` must be positive
semidefinite and is supplied in the upper-triangular CSC convention; MathOptInterface's
`ScalarQuadraticFunction` convention is used unchanged, so a diagonal term
`ScalarQuadraticTerm(a, x, x)` contributes `(1/2) a x^2` and an off-diagonal
term `ScalarQuadraticTerm(a, x, y)` contributes `a x y`.

## Numeric types

`Float64` and `Float32` are both supported and both tested. Construct
`JuliaQOCO.Optimizer{Float32}` for single precision, or pass `Float32` data to
the native interface; the sparse index type is independent and may be narrowed
to `Int32` as well.

Every tolerance, regularization size and division guard is derived from
`eps(T)` rather than written as a literal, because a fixed constant is not a
safe default across precisions. Concretely, an absolute tolerance of `1e-7` is
*below* `eps(Float32)` and so can never be met, and a regularization shift of
`1e-8` added to a diagonal entry of order one vanishes entirely in single
precision. The `Float64` defaults are unchanged: each rule is a floor at the
tuned double-precision value plus a term proportional to `eps(T)` that only
takes over in lower precision.

| setting | `Float64` | `Float32` |
| --- | ---: | ---: |
| `abstol`, `reltol` | 1e-7 | 3.45e-5 |
| `abstol_inacc`, `reltol_inacc` | 1e-5 | 3.45e-3 |
| `kkt_static_reg`, `kkt_dynamic_reg` | 1e-8 | 2.31e-4 |
| `iter_ref_tol` | 1.49e-8 | 1e-5 |

Accuracy follows the arithmetic: on the test fixtures, `Float64` reaches
residuals around `1e-13` and `Float32` around `1e-6`, and in both cases the
reported status is backed by the same independently checked original-unit
quality. Single precision is not a way to get double-precision answers faster;
it is a way to solve to single-precision accuracy in half the memory.

Other `AbstractFloat` types will construct and run, since the rules above are
written in terms of `eps(T)`, but only `Float64` and `Float32` are covered by
the test suite. `BigFloat` in particular will not benefit from tighter
tolerances than the `Float64` floors allow.

Supported constraints are `VariableIndex` and `ScalarAffineFunction` in
`EqualTo`, `LessThan`, `GreaterThan` and `Interval`, and `VectorOfVariables`
and `VectorAffineFunction` in `Zeros`, `Nonnegatives` and `SecondOrderCone`.
Second-order cones must have dimension at least two.

## Repeated solves

Build one direct model and update the functions in place:

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

### What updates and what rebuilds

| Change | Effect |
| --- | --- |
| Coefficient, right-hand side, bound or objective-coefficient change | Numerical update |
| Whole scalar or vector affine function replacement, same dimension, support inside the allocated pattern | Numerical update |
| A reserved coefficient going to zero, and later becoming nonzero again | Numerical update |
| Objective sense change with an affine objective | Numerical update |
| A genuinely new nonzero coordinate, a changed dimension, a changed cone or bound pattern | One symbolic rebuild on the next solve |
| Adding or deleting variables or constraints | One symbolic rebuild on the next solve |
| Changing `scaling_mode`, `ruiz_iters`, `kkt_static_reg` or `soc_expansion_threshold` | One symbolic rebuild on the next solve |
| Changing tolerances, `verbose`, `kkt_dynamic_reg`, `warm_start_mode`, `max_iters` or the time limit | Numerical update only |

A numerical update reuses the solver object, the AMD ordering, the symbolic
factorization, the factor storage and the coordinate maps. Verify this in a
test with `MOI.RawSolver()` identity plus `MOI.RawOptimizerAttribute("rebuild_count")`,
not with an iteration count.

**Structural support is preserved separately from the current values.** A
coefficient that is currently zero still owns its slot, so a whole-function
replacement that omits it stays an update.

Every numerical update is validated before anything is committed. A rejected
update, such as one containing a nonfinite value, an out-of-range index or a
Hessian that is no longer positive semidefinite, leaves the editable model, the
raw values, the dirty queues and the native matrices in agreement.

### Scaling modes

| `scaling_mode` | Meaning |
| --- | --- |
| `:none` | No equilibration. |
| `:once` | Ruiz equilibration is computed at construction and frozen. Best for repeated fixed-pattern solves; the default. |
| `:recompute` | Equilibration is recomputed whenever matrix data changes. More robust to data whose magnitudes drift a long way over a run, at the cost of redoing the sweep. |

Indexed entry updates require `:none` or `:once`; with `:recompute` the whole
transaction is pushed through `update_data!` so that the new objective vector
participates in the objective scale of the same sweep.

### Warm-start modes

| `warm_start_mode` | Meaning |
| --- | --- |
| `:none` | The previous solution is never reused automatically. An explicit start supplied by the caller is still honoured. |
| `:primal` | The primal point is reused; the slack is rebuilt from it and a fresh, strictly interior dual pair is used. |
| `:primal_dual` | All four components are transformed into the current scaling and reused, with repair back into the cone interior. The default. |
| `:adaptive` | As `:primal_dual`, but the transformed point is also checked for centrality; a badly uncentered point has its stale duals dropped and primal-only reuse tried once before the start is abandoned. |

A reused start is accepted only if it is finite, strictly inside the cone after
repair, and within a bounded relative residual of the new problem measured in
original units. A rejected start falls back to a cold start. If an accepted
start leads the interior-point method into an early stall, the solve is retried
once from cold under the same iteration and time budget, and the better of the
two results is kept.

Only a result that was actually published, is finite and lies in the cone is
cached for the next solve. Nothing else is.

## Statuses, results and what they mean

Residuals and objectives are reported in the **original problem units**, not in
the internally scaled ones. Concretely, with the internal scaling
`P_hat = k D P D`, `c_hat = k D c`, `A_hat = E A D`, `G_hat = F G D`, the
reported quantities are

```text
objective  = (1/2) x' P x + c' x
pres       = max(|A x - b|_inf, |G x + s - h|_inf)
dres       = |P x + c + A' y + G' z|_inf
gap        = s' z
```

with `P` the mathematical Hessian, excluding the static regularization shift
that the factorization adds. Every success status is backed by these
quantities: `MOI.OPTIMAL` means the largest of the three residual-to-tolerance
ratios is at most one *and* the returned iterate lies in the cone and is
finite.

The reported metrics always describe the vectors that are returned. When the
solver falls back to an earlier iterate, that iterate's objective, residuals,
complementarity and cone validity are recomputed before anything is published.
`MOI.BarrierIterations` reports the interior-point iterations actually
performed, which is deliberately not the same number as the index of the
iterate that was published.

When no usable result exists at all, `MOI.ResultCount()` is zero and both
`MOI.PrimalStatus()` and `MOI.DualStatus()` are `MOI.NO_SOLUTION`. An
all-zero or stale vector is never presented as a newly computed solution.

**No infeasibility certificates.** This solver does not implement a
homogeneous embedding. `MOI.ITERATION_LIMIT`, `MOI.NUMERICAL_ERROR`,
`MOI.TIME_LIMIT` and an inaccurate solution are all statements about the solve,
not proof that the problem has no solution. An infeasible subproblem typically
costs the full iteration budget before it is reported, so a sequential convex
programming loop that can generate infeasible subproblems should set
`max_iters` or `MOI.TimeLimitSec()` deliberately. `MOI.ObjectiveBound`,
`MOI.DualObjectiveValue`, `MOI.RelativeGap` and basis statuses are not
produced and are declined rather than approximated.

## Convexity validation

`convexity_check` controls how the quadratic objective is verified:

- `:auto` (default) verifies on construction and on every update that changes
  the Hessian, using the cheapest conclusive test for the pattern at hand. A
  diagonal Hessian, the common case for sequential convex programming
  penalties, costs one pass over the nonzeros. Otherwise a dense symmetric
  eigenvalue decomposition is used below `convexity_dense_limit`, and a
  Cholesky factorization of the Hessian plus a small shift above it, which
  proves the same semidefinite bound at far lower cost.
- `:none` means the caller guarantees convexity. No test is run, and no claim
  of verified convexity is made. Use this only when the Hessian is convex by
  construction.

A nonconvex Hessian slipped past `:none` is not silently certified: the
regularized factorization may still succeed, but the stopping criteria are
checked against the true, unregularized residuals in original units, so the
solve fails rather than returning a false optimum.

## Large second-order cones

A single big second-order cone, such as a global Euclidean trust region over a
whole trajectory, would otherwise dominate the solve: the dense Nesterov-Todd
block costs `q(q+1)/2` entries and, measured on a horizon-40 control problem,
88% of the total solve time. Cones of dimension at least
`soc_expansion_threshold` (16 by default) instead use an exact sparse
expansion with two auxiliary variables per cone, which is `O(q)` in both
storage and update cost.

The expansion reproduces the original Newton block exactly - it is an
identity, not a relaxation - and the tests assert that it gives the same
iterates, the same iteration count and the same accuracy as the dense block.
Measured on a control fixture with one trust-region cone:

| cone dimension | dense | expanded | speedup | dense `nnz(L)` | expanded `nnz(L)` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 0.092 ms | 0.040 ms | 2.3x | 2268 | 861 |
| 187 | 4.31 ms | 0.41 ms | 10.5x | 27567 | 5043 |
| 247 | 6.89 ms | 0.42 ms | 16.5x | 44706 | 6723 |
| 367 | 18.8 ms | 0.65 ms | 29x | 89784 | 10083 |

The default threshold sits above the measured crossover, near dimension ten,
so the many small thrust-style cones of a typical trajectory problem keep the
compact dense block. Set `soc_expansion_threshold` to `typemax(Int)` to
disable the expansion.

## Time budget

`MOI.TimeLimitSec()` sets a wall-clock budget, checked at iteration boundaries
using a monotonic clock. One factorization call is not preemptible, so this is
a budget, not a hard real-time deadline. Any automatic cold-start retry shares
the same budget. Reaching the budget yields `MOI.TIME_LIMIT` together with the
best iterate found, which is finalized and graded like any other result.

## Diagnostics

Available through `MOI.RawOptimizerAttribute`: `rebuild_count`,
`symbolic_rebuild_count`, `structure_generation`, `last_rebuild_reason`,
`last_commit_time_sec`, `commit_count`, `regularized_entries` and
`dynamic_regularizations`. The cached solver's `Solution.profile` additionally
separates structural rebuilds, numerical refactorizations, regularized pivots,
factorization retries, and warm starts accepted, repaired, rejected and
retried. Note that "native solve time" (`MOI.SolveTimeSec()`) covers the
interior-point solve only; `last_commit_time_sec` covers the numerical commit,
and neither includes the caller-side cost of modifying the model.

Profiling is observational. Turning `profile` on does not change any numerical
result; the tests assert bit-for-bit equality between profiled and unprofiled
runs.

## Tests and benchmarks

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The full test run includes a broad MathOptInterface conformance sweep and takes
several minutes. During development, run a subset:

```sh
julia --project=. test/runtests.jl native numerics
julia --project=. test/runtests.jl types      # Float64 and Float32 coverage
julia --project=. test/runtests.jl moi
```

Benchmarks live under `benchmark/` and are opt-in:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/runbenchmarks.jl            # full suite
julia --project=benchmark benchmark/runbenchmarks.jl --quick    # smoke run
julia --project=benchmark benchmark/baseline_comparison.jl      # versus the audited revision
julia --project=benchmark examples/scp_jump_reuse.jl            # smoke example
```

Results are written to `benchmark/results/`, which is build output rather than
package source. See `docs/internals.md` for the numerical conventions and
`benchmark/results/report.md` for the full measurements and their limitations.

Accuracy-matched against the revision this work started from, on the full
update-and-solve replay sequence, with both sides graded by the same
independent oracle at a relative tolerance of `1e-5`:

| fixture | largest SOC | before | after | speedup |
| --- | ---: | ---: | ---: | ---: |
| small | 45 | 0.107 ms | 0.074 ms | 1.45x |
| medium | 187 | 5.33 ms | 0.489 ms | 10.9x |
| large-soc | 247 | 9.81 ms | 0.757 ms | 13.0x |

Every step of every fixture passed the quality gate on both sides, and the
worst relative residual improved or stayed equal in all three.

JuliaQOCO is based on the pure-Julia numerical method described in the QOCO
paper. The vendored sparse direct factorization is derived from
`oxfordcontrol/QDLDL.jl` under Apache-2.0; see `licenses/QDLDL-LICENSE`.
