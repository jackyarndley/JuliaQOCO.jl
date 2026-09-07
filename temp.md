# Codex task: make JuliaQOCO reliable and fast for repeated SCP subproblems

## Objective

Work in `jackyarndley/JuliaQOCO.jl`. Implement, test, benchmark, and document improvements that make this native Julia quadratic-objective conic solver a reliable, low-overhead alternative for sequential convex programming (SCP).

Optimize **time to a numerically acceptable SCP result**, not merely reported solver time or iteration count. Correctness, reproducibility, readable code, and repeated-solve performance are all requirements. Do not stop at an assessment or a proposed patch: carry out the implementation and verification available in your environment.

This task is based on a source audit of `main` at commit:

`c2725866335a51a54b10450726c22c05c4127dc9` (6 August 2026).

The older `temp` branch is not the audit baseline. Inspect the current checkout, repository instructions, working-tree changes, and current branch before editing. Treat the audited commit as an evidence anchor, not an instruction to reset the repository. Preserve user changes and incorporate subsequent fixes rather than reintroducing old code.

**Evidence limitation:** the initiating audit inspected the source and checked the scaling algebra, but did not execute the Julia package or measure its speed. Findings described below as source defects require executable regression tests. Performance opportunities are hypotheses until measured. Do not present the audit as a passing or failing test run.

## Architectural and style constraints

- Keep the numerical implementation native Julia. C QOCO and Clarabel are independent comparison solvers, not replacement engines for JuliaQOCO.
- Preserve `JuliaQOCO.Optimizer` as the supported public entry point. The repository deliberately tests that `CoreSolver` and its internal solve/update functions are not exported. Private benchmark access does not require a new public API.
- Keep JuMP/MOI integration, including `direct_model`, usable. Optimize the wrapper as well as the core; do not solve the performance problem by requiring users to abandon the supported interface.
- Prefer compact, concrete data structures and ordinary readable functions. Reuse the existing workspace, assembly cache, dirty queues, and sparse maps. Do not add another solver framework, provider hierarchy, general plugin system, or collection of one-use abstraction layers.
- Share actual numerical logic, especially result finalization and tolerance calculations. Keep profiling observational: it must not change numerical behavior. Do not duplicate a whole solver implementation to add an optimized path.
- Preserve useful comments explaining equations and data conventions. Remove dead code and obsolete internals rather than retaining parallel legacy implementations. Do not pursue a line-count target at the expense of clarity or correctness.
- Preserve existing license notices and the vendored QDLDL attribution. Additional borrowed code requires its appropriate attribution and license.
- Keep benchmarking/reference-solver dependencies out of the runtime dependency set. Respect the declared Julia compatibility (`julia = "1.10"` at the audited revision) unless a change is explicitly justified and documented.

## Preserve what already works

Do not propose the following as missing: the repository already has cached solver reuse, AMD ordering, symbolic factorization reuse, a cached numeric QDLDL pattern, in-place CSC updates, transpose/KKT/factor index maps, dirty queues, frozen scaling, automatic and manual warm starts, best-iterate buffers, analytical SOC line search with a fallback, compact W/Winv multiplication, profiling, and a repeated-solve example.

Distinguish **symbolic reuse** from **numeric factor reuse**. Numerical NT scaling normally changes the KKT matrix each IPM iteration, so numerical factorization is still needed. A no-change re-solve that terminates immediately is useful, but does not demonstrate the cost of an SCP iteration with changed linearization data.

## Priority and implementation order

| Priority | Work | Required outcome |
| --- | --- | --- |
| P0 | Original-unit stopping criteria, zero-objective scaling, final-iterate/result consistency | Trustworthy status, objective, residuals, and returned iterate |
| P0 | Update validation and objective-sense correctness | Updated problems represent exactly the requested mathematics |
| P1 | Warm-start and rescaling semantics, numerical safeguards, MOI result behavior | Reliable repeated solves and controlled failure/recovery |
| P1 | Linear-time function updates, appropriate convexity checks, independent benchmarks | Measured reduction in real update-and-solve overhead |
| P2 | Large-SOC sparse expansion and other structure exploitation | Retain only demonstrated, accuracy-preserving improvements |
| P2 | Broader precompilation, time limits, documentation, regression automation | Predictable and maintainable integration |

Implement the P0/P1 work before undertaking a large new linear-system representation. Keep commits or changesets logically separated. Do not leave verified correctness fixes blocked behind speculative performance experiments.

## 1. Establish a reproducible baseline and independent numerical oracle

Read the complete current implementation and tests before editing. Map the execution path:

`MOI model modification -> assembly/dirty queues -> numerical commit -> scaling/warm start -> residuals/NT/KKT/IPM -> finalization -> MOI results`.

Run the existing tests, retaining the exact command, Julia version, package versions, commit, and output. Record existing failures without changing expected tolerances to hide them. Save the baseline benchmark results before optimization, but validate their numerical quality independently: an old `OPTIMAL` status is not a sufficient quality gate.

Implement a small, transparent **test/benchmark-only oracle** using the original, unscaled problem data. It must not call the production residual or stopping functions it is meant to validate. For small cases use ordinary dense linear algebra; for larger cases use independent sparse products. Check:

- objective value, including the correct MOI sense and constant;
- equality and conic equation residuals;
- dual stationarity;
- primal and dual cone membership;
- complementarity;
- finiteness of all returned vectors and reported metrics;
- consistency between reported metrics and the returned iterate.

Generate small feasible LP/QP/SOCP instances with known solutions or constructed KKT points. Include diagonal and coupled positive-semidefinite Hessians, singular PSD Hessians, equality-only cases, mixed cones, degenerate/near-boundary cases, and substantial but representable scaling differences. Compare to an independent solver where useful. With nonunique optima, compare objective and KKT quality rather than requiring identical primal vectors.

## 2. Correct stopping criteria and scaling algebra

### Evidence locations

`src/kkt.jl`: `compute_kkt_residual!`, `check_stopping!`.

`src/equilibration.jl`: `ruiz_equilibration!`, `unscaled_solution!`.

`src/utils.jl`: `weighted_dot`, `weighted_inf_norm`.

At the audited revision:

1. `cinf` in `check_stopping!` is computed from `work.x`, not the objective vector `data.c`.
2. The slack norm uses `Fruiz` instead of its inverse.
3. The complementarity calculation uses `weighted_dot(work.s, work.z, Fruiz)`, introducing an erroneous squared row-scaling factor.
4. `work.xPx` uses `weighted_dot(work.x, work.xbuff, Dinvruiz)`, which is not the original quadratic form.
5. Objective terms in the relative-gap denominator are not consistently converted to original objective units.
6. Comparisons should use the magnitude of complementarity, with separate finiteness and cone-validity checks, rather than allowing negative or invalid values to imply success.

### Required derivation and implementation

Document the actual conventions. With positive diagonal scaling matrices D, E, F, objective scale k, and hats denoting internal scaled data:

```text
P_hat = k D P D                 (excluding numerical static regularization)
c_hat = k D c
A_hat = E A D,   b_hat = E b
G_hat = F G D,   h_hat = F h

x = D x_hat
s = F^(-1) s_hat
y = E y_hat / k
z = F z_hat / k
```

The following identities must hold:

```text
s' z = dot(s_hat, z_hat) / k
x' P x = dot(x_hat, P_hat * x_hat) / k
norm(c, Inf) = norm(D^(-1) * c_hat, Inf) / k
norm(s, Inf) = norm(F^(-1) * s_hat, Inf)
```

The stored `data.P` currently includes a numerical diagonal shift. Exclude that shift from the mathematical objective and original KKT residual; do not accidentally solve or report a regularized objective as the requested objective.

Compute original-unit residuals and the corresponding reference norms consistently:

```text
r_eq   = A*x - b
r_cone = G*x + s - h
r_dual = P*x + c + A'*y + G'*z

p_ref = max(norm(A*x,Inf), norm(b,Inf), norm(G*x,Inf),
            norm(h,Inf), norm(s,Inf))
d_ref = max(norm(P*x,Inf), norm(c,Inf), norm(A'*y,Inf),
            norm(G'*z,Inf))
```

Use an explicitly documented, consistent objective-unit reference for complementarity. The usual expression `-.5*x'*P*x - b'*y - h'*z` can be useful in the stopping scale, but is not automatically a rigorous dual lower bound for an arbitrary nonstationary iterate. Do not expose a claimed certified dual objective merely by computing this expression. Treat additive objective constants consistently and avoid using arbitrary constant offsets to disguise inaccurate solves.

Use shared accuracy calculations for regular stopping, inaccurate stopping, best-iterate ranking, and final status assessment. A normalized maximum of the individual residual-to-tolerance ratios is preferable to the current absolute-only best metric. Account for both absolute and relative tolerances and the requested precision.

Avoid allocating unscaled vectors on every iteration. Derive the appropriate weighted products from the internal representation, and verify them against the independent oracle.

### Regression tests

Test non-unit and nonuniform D, E, F, and k. SOC row scaling must remain cone-preserving. Include transformations of the same mathematical problem and verify compatible solution quality after undoing the transformation.

A useful algebraic regression is a uniform cone scale `F = 1e-4 I`, `k = 1`: the current weighted-gap formula yields `1e-8` times the true complementarity. A true gap of `0.1` therefore becomes `1e-9` in that formula. This is an algebraic illustration, not a claim that a particular complete solver run has already been reproduced.

Also test that scaling does not change the meaning of the stopping threshold and that status never indicates success merely because an intermediate norm became NaN, Inf, or spuriously zero.

## 3. Make scaling and numerical guards safe

### Zero and tiny objective data

At the audited revision, `ruiz_equilibration!` computes an objective norm g and then calls `safe_div(1, g)`. `safe_div` returns `floatmax(T)` for denominators with magnitude at most `1e-15`.

For `P = 0` and `c = 0`, this can repeatedly multiply the objective scaling by `floatmax(T)` and overflow k. A zero-objective feasibility problem must not cause this behavior.

Handle zero objective norms explicitly, normally with neutral objective scaling. Bound scaling changes where necessary, retain finite positive inverse scales, and test zero/very small/very large data. Select a representative common scaling for an entire SOC rather than blindly relying on only its head row when that is poorly representative. Preserve the SOC itself under row scaling.

Audit every use of `safe_div`. Do not replace legitimate small denominators with an enormous positive sentinel, erase signs, or mask invalid numerical states. Different uses require different handling: safe scaled evaluation, explicit zero cases, strict interior repair, or a controlled numerical failure. Keep the policy local and mathematically meaningful rather than creating a complex generic arithmetic layer.

Replace NaN-only checks with appropriate finite checks at numerical boundaries. In particular, inspect warm starts, search directions, pivot values, inverse pivots, residuals, and final solutions. Reject nonfinite settings and input data before mutating committed solver state.

### SOC robustness and line search

The analytical SOC line search already exists. Improve and test it instead of adding a second unrelated implementation.

Check determinant/norm evaluation near the SOC boundary and for large/small magnitudes. The expression `u0^2 - sum(utail.^2)` is vulnerable to cancellation and overflow. Use appropriately scaled numerical formulas and explicit cone-interiority checks. Do not silently mask a materially invalid cone point by clamping its determinant to zero.

The current bisection fallback uses five iterations on `[0,1]`; it can return zero when a positive feasible step is smaller than `1/32`. Replace this failure mode with a safeguarded feasibility check and a justified bracketing/backtracking policy. Test tiny positive feasible steps, near-linear quadratic equations, both directions of the SOC boundary, zero directions, and mixed orthant/SOC problems.

Make the fraction-to-boundary rule consistent: permit a unit step when it satisfies the intended interior margin, rather than unnecessarily damping every unrestricted direction. Verify the analytical and fallback paths against a high-accuracy reference in small tests.

## 4. Finalize one coherent result on every exit

### Evidence locations

`src/solver.jl`: `_record_best_iterate!`, `_restore_best_iterate!`, `_solve_fast_loop!`, `_solve_profiled_loop!`, `_solve!`.

At the audited revision, restoring a best iterate only restores x, s, y, and z. `unscaled_solution!` does not refresh the objective or residuals. Reported metrics can therefore describe a different iterate from the returned vectors. The last computed IPM step is also not assessed before the iteration-limit restoration.

Implement one clear finalization path that:

1. Assesses the most recently computed iterate, including the final permitted step.
2. Selects a valid current/best iterate using the shared normalized quality metric.
3. Recomputes the mathematical objective, residuals, complementarity, and cone validity for the selected iterate.
4. Unscales and publishes a mutually consistent result.
5. Assigns an honest termination reason and result availability.
6. Updates the warm-start cache only under an explicit quality/validity policy.

Record iteration accounting clearly. The number of iterations performed need not equal the index of a restored iterate; do not make one field ambiguously mean both.

Cover normal convergence, inaccurate convergence, small-step stall, iteration limit, numerical failure, time limit if implemented, and initialization failure. An initialization failure with no finite valid result must not expose an all-zero or stale vector as a newly computed solution.

Test `max_iters = 1`, deliberately forced late failure, restoration of an earlier iterate, and equality of profiled/nonprofiled numerical results. Compare every returned objective and residual against the independent oracle.

## 5. Correct fixed-pattern updates and MOI semantics

### Objective-sense bug

Inspect `_objective_data`, `_queue_objective_function!`, and `_queue_target!` in `src/moi_wrapper.jl`.

At the audited revision, a quadratic target already contains the objective-sense multiplier. Whole quadratic-objective replacement also multiplies a coefficient by `cache.objective_sign` before sending it to the target, applying that sign twice. This is wrong for concave maximization converted to convex minimization. Scalar quadratic-coefficient modifications follow a different path and must agree with whole replacement.

Add a regression such as maximizing `-x^2 + x` on a suitable bounded interval, then replacing the objective with `-2x^2 + x`. The updated optimum is `x = 1/4`; the internal minimized Hessian for the updated scalar quadratic term is positive. Compare initial construction, whole replacement, scalar coefficient changes, and a freshly constructed optimizer. Respect MOI's diagonal quadratic coefficient convention rather than introducing a spurious factor-of-two fix.

Test objective constants, affine terms, minimization, maximization, feasibility sense, and sense changes. Each conversion should happen exactly once.

### Checked and atomic numerical updates

Audit every route: scalar coefficient modifications, MultirowChange, full scalar/vector function replacements, RHS/bound changes, quadratic updates, and internal bulk/indexed updates.

Validate finite numerical values, dimensions, variable references, and index bounds before entering unchecked kernels. Preserve a valid committed state on rejected updates. A rejected modification must not leave the editable MOI model, raw values, dirty queues, and native matrices disagreeing.

Use one validation/commit boundary rather than repeating expensive validation per coefficient. Distinguish cheap required validation from expensive optional convexity analysis. All fields in a matrix/vector update transaction should be considered before recomputing scaling, including changes to c that affect objective scaling.

Handle bounds correctly: a nonbinding infinity is not the same as NaN or an infinity in the contradictory direction. The audited assembly path skips any nonfinite scalar inequality bound. Do not silently turn invalid/contradictory input into an absent constraint. Report unsupported invalid input appropriately without claiming an infeasibility certificate the solver has not computed.

Resolve zero-dimensional cone behavior: the native validation allows q=0, while cone kernels assume a head entry. Either normalize such blocks out safely or reject them consistently at the relevant boundary.

### Canonicalization and structural zeros

MOI functions can contain duplicate terms. Canonicalize or accumulate them consistently at the MOI boundary; do not confuse that with accepting malformed duplicate row entries in native CSC storage.

Preserve the **allocated structural support**, separately from the current nonzero values and the current canonical MOI expression. Coefficients becoming zero or later becoming nonzero again within reserved slots must not require a symbolic rebuild. Whole-function replacement that omits a currently zero reserved term should remain an update when its active support stays within the allocated pattern.

A genuinely new nonzero coordinate, changed dimensions, or a changed cone/bound structure may require one rebuild on the next solve. Test solver identity, symbolic structure identity, and numerical equivalence to a cold reconstruction, not just the rebuild counter.

### Convexity policy

The audited code performs dense eigenvalue checks only for quadratic problems up to 512 variables, can repeat them on full objective replacements, and bypasses equivalent checks on coefficient-update paths. The size cutoff also depends on the largest MOI variable ID in one path.

Replace this with a consistent, documented policy. Cover semidefinite as well as definite matrices. Avoid an O(n^3) dense eigenvalue calculation every time an SCP objective is replaced. Use cheap diagonal/block structure checks where valid and a justified general check where requested. An explicit caller-guarantee option may be reasonable, but unchecked input must not silently be represented as verified convex because n exceeded an arbitrary threshold.

Do not use sign-forced regularized LDL inertia as a proof that the original Hessian is PSD. A plain Cholesky test without appropriate treatment of semidefinite cases is also insufficient. Test small/large dimensions, deleted variable IDs, indefinite Hessians, and all update routes.

### Remaining MOI behavior

Fix and test `MOI.Silent` changes after a cache already exists: changing optimizer settings must actually reach the cached solver. Tolerance changes should not unnecessarily rebuild the symbolic factorization. Reconsider whether changing dynamic regularization alone requires a structural rebuild.

Check result-index semantics, result count, inaccurate solution statuses, dual signs, intervals, and result invalidation after modifications. Report primal/dual feasibility status according to independently supported quality, not merely the broad termination label.

Audit claimed start-attribute support. The existing explicit implementation consumes variable primal starts; implement supported dual-start attributes correctly or explicitly decline them rather than merely storing unused data in a fallback model.

Preserve InterruptException through every wrapper layer. The audited outer `MOI.optimize!` catch can swallow it even though the inner solve rethrows it. Do not convert programming errors indiscriminately into routine numerical failures; keep expected numerical-failure handling explicit and debuggable.

## 6. Make warm starts correct before making them more aggressive

### Rescaling bug

In `src/updates.jl`, `update_matrix_data!` converts a cached scaled warm start to unscaled vectors before recomputing Ruiz scaling. `_apply_warmstart!` then rescales x and y, but rescales an unscaled z only in its manual-start branch. An automatic warm start after recomputation can therefore retain z in the old scaling.

Use one unambiguous coordinate-conversion path for all four components. Reconstructing slack as `h - G*x` is a separate policy choice from converting dual coordinates. Verify x, s, y, and z independently when D, E, F, and k change.

### Mode semantics

At the audited revision, `:adaptive` and `:primal_dual` share the same implementation path and broad residual gate. `:primal` clears duals only in some branches. Disabling warm starts does not necessarily prevent an already active cache from being applied to the next solve.

Define and enforce the modes explicitly:

- `:none`: do not use an automatic previous solution on the next solve. Document how an explicit manual start interacts with this setting.
- `:primal`: reuse primal information and construct valid interior slack/dual initialization; do not accidentally retain old duals in equality-only or unchanged-data cases.
- `:primal_dual`: transform and reuse both components with justified repair.
- `:adaptive`: make a real, inexpensive quality-based choice among useful reuse and cold initialization, or replace the misleading behavior with a clearly documented alternative consistent with repository compatibility requirements.

Use original-unit normalized residuals, strict cone-interiority, complementarity, and sensible centrality information where useful. Avoid an arbitrary scaled residual threshold as the entire acceptance policy. Interior margins should be relative to cone scale and floating-point precision, not always a fixed `1e-4` shift.

If a reused start causes an early numerical stall, consider one bounded cold-start retry under the same solve budget. Instrument whether the warm start was accepted, repaired, rejected, or retried. Do not repeatedly restart indefinitely, and do not claim that warm starts always reduce IPM iterations.

Test vector-only updates, matrix updates, frozen/recomputed scaling, unchanged problems, changing modes after a solve, equality-only cases, large trust-radius reductions, large penalty changes, partial manual starts, and invalid starts. Compare updated results with cold reconstruction at matched accuracy.

## 7. Define the KKT/refinement target and harden factorization

The original Newton system, the statically regularized factorization matrix, and the dynamically modified LDL factors are different objects. Make that distinction explicit.

At the audited revision, `kkt_multiply!` includes the regularization already stored in P but omits the additional equality/cone diagonal regularization used by the factor matrix. This is not a consistent implementation of either the fully regularized matrix or the fully unregularized Newton matrix.

Derive the intended refinement target. Normally the regularized factor is used to approximately solve a correction equation whose residual is measured against the original Newton operator. Verify every block, including the subtraction of the primal static shift. Do not blindly add missing shifts to make two matrices equal if doing so changes the intended refinement algorithm.

Test KKT products and solves against explicitly assembled matrices in small cases. Measure backward error and handle stagnating/diverging refinement explicitly. Keep the same factor for predictor and corrector solves and for refinement corrections where mathematically valid.

In `src/internal_qdldl.jl`, both the initial and cached numeric factorization paths must detect nonfinite pivots and invalid inverses, not only exact zero pivots. In `update_nt_block!`, restrict retries to numerical failures and preserve interrupts. Bound regularization escalation, count actual regularized pivots separately from retry attempts, and verify the recovered solution with the original Newton operator.

Test factorization and refactorization over deterministic sequences against fresh factorizations and dense references. Include zero-crossing matrix entries within a fixed pattern, badly scaled systems, rank-deficient equalities, singular PSD P, and failed factorization recovery. A quasidefinite regularization policy must not turn an invalid nonconvex model into a falsely certified optimum.

## 8. Remove avoidable repeated-model overhead

### Linear-time support checks

The audited `_same_scalar_term_support`, `_same_vector_term_support`, and `_same_quadratic_term_support` compare every old term against every new term. This is O(t^2), which is especially undesirable for a large vector-valued dynamics constraint.

Use the existing cached coordinate/slot information with an O(t) expected-time membership/marking approach, or an O(t) merge on a maintained canonical ordering. Handle duplicate accumulation, reordered terms, and reserved zero slots correctly. A hash alone is not a correctness proof of equal support.

Do not allocate a new Set/Dict proportional to the entire model for every update. Reuse a compact scratch buffer or generation marks when needed, but avoid creating a second cache hierarchy.

### Changed-only commit and batching

The audited whole-function update first queues zeros for old terms and then all new terms; whole-objective updates zero and queue the entire c vector and all quadratic targets. Reduce redundant passes and work where the support is already known.

Retain the existing dirty-queue mechanism. Queue actual changed values, avoid avoidable duplicate dictionary lookups, and keep sparse modifications proportional to the touched data in the native commit path. For full updates, a contiguous O(nnz) pass can be better than thousands of indexed dispatches; select a small, evidence-backed approach rather than a complicated adaptive subsystem.

Measure model mutation separately from native commit. `MOIU.UniversalFallback` remains an editable model representation, so a faster native indexed kernel does not by itself prove that thousands of JuMP coefficient changes are cheap. Profile actual scalar, multirow, and whole-function caller paths before replacing the representation.

Check allocations with warmed functions and interpolated benchmark arguments. Distinguish caller-created MOI term arrays, model-storage allocations, commit allocations, factorization allocations, and output extraction. Do not demand that allocating `value.(x)` or constructing a new expression be allocation-free; report those costs separately.

### Other small measured improvements

The core already uses concrete factor/workspace types and in-place kernels. Do not describe generic type stability or preallocation advice as a discovered defect.

Remove unused calculations in `compute_nt_scaling!`, including the loop computing unused `wval`/`winvval`, after checking their lack of side effects. The compiler may already eliminate them, so do not invent a runtime improvement.

The synthetic `JuMPSCPCase` uses untyped fields for several containers. Give benchmark/example data an appropriate concrete representation where doing so prevents the harness from polluting measurements. Do not confuse that example issue with the numerical core's types.

For equality-only QPs, assess whether the generic cone predictor/corrector loop does unnecessary repeated work. A small direct KKT solve path may be useful if it remains simple, handles singularity honestly, and is independently verified. It is not the main SCP/SOCP performance claim.

## 9. Investigate large-cone KKT sparsity, after the baseline is trustworthy

### Main structural opportunity: sparse SOC expansion

`kkt_nt_nnz` and `construct_kkt` allocate a dense upper-triangular NT block for every SOC. The per-cone storage is `q*(q+1)/2`, although W and Winv multiplication already have compact implementations.

For q=1000, that is 500,500 entries in just one triangular block, before maps, multiple value arrays, indices, and factor fill. This matters for a large global Euclidean trust-region cone, not merely for many thrust cones of dimension three or four.

Prototype an exact sparse auxiliary-variable/low-rank expansion of the SOC KKT block. Requirements:

- Derive the Schur complement algebra explicitly; it must reproduce the original Newton block, not approximate or replace the optimization problem.
- Preserve a fixed structural pattern and cached symbolic factorization across IPM iterations and SCP updates.
- Derive signs and regularization for the augmented system rather than copying the old Dsigns vector blindly.
- Correctly map RHSs, recover original search directions, and evaluate refinement against the original operator.
- Compare dense and expanded block products/directions over randomized interior cone points and near-boundary cases.
- Retain the compact dense path for small cones when it is faster/simpler. Choose any threshold from measurements, not from an assumed universal rule.
- Report both input KKT nonzeros and actual factor fill. O(q) block storage does not guarantee O(q) total factorization work.

Do not replace one global L2 trust region with separate stagewise bounds just to shrink the cone: that changes the feasible set and the SCP method unless deliberately reformulated with proven equivalence.

### Secondary candidates, only if profiles justify them

Assess elimination of simple one-variable orthant rows, such as bounds. If an orthant block is `-D_l`, eliminating its dual direction contributes `G_l' * inv(D_l) * G_l` to the primal block and the corresponding transformed RHS. For one-variable rows this contributes only to the diagonal; general rows can create substantial fill. Verify signs, regularization, RHS mapping, and recovery. Do not broadly condense all inequalities or all trajectory states without measuring the fill and conditioning costs.

Compare AMD with sensible stage-aware orderings using actual `nnz(L)`, solve/refactor time, and stability. Do not add an abstract backend registry to try two orderings.

A later fixed-pattern Julia arithmetic schedule could take inspiration from QOCOGEN, but the repository already caches the numeric sparsity traversal. Benchmark incremental benefit beyond that baseline. Avoid unrolling an entire large horizon into generated code without measuring compilation time, code size, cache effects, and first-use latency.

Do not begin with GPU support, mixed precision, a new homogeneous embedding, or a wholesale solver rewrite. Keep experimental changes separate and retain them in production only when numerical tests and representative end-to-end measurements support them.

## 10. Replace misleading performance evidence with matched workloads

### Issues in the current benchmark

The existing benchmark is valuable but insufficient as the final performance claim:

- `examples/scp_jump_reuse.jl` is a synthetic sequence of linear-dynamics SOCPs, not an actual nonlinear SCP loop with relinearization, nonlinear acceptance checks, virtual controls, and penalty updates.
- Mutable benchmark cases evolve between samples. The whole-function updater multiplies old coefficients, while another update path reconstructs coefficients from a fixed base. These are not equivalent sequences.
- A no-change solve can terminate without IPM work.
- `solver_reused = (iters > 0)` in one output line does not establish solver identity.
- Timing output is not gated by independent primal/dual quality checks, and there is no matched reference-solver comparison.

Fix these points rather than deleting the useful existing smoke example.

### A. Deterministic replay benchmark

Precompute a fixed sequence of original problem data and structural patterns. Start each solver/configuration from the same initial instance, then replay identical numerical updates. Reset between repetitions; do not let BenchmarkTools tuning determine a different sequence for each method.

Include these workloads:

| Workload | What it isolates |
| --- | --- |
| Fresh setup and solve | Construction, validation, ordering, symbolic factorization, solve |
| Unchanged re-solve | Reuse overhead only; label it accordingly |
| c/b/h updates | Cost/RHS/bound changes without structural change |
| Full A/G updates | Typical changed linearization |
| P plus vector updates | Changed quadratic penalties or approximations |
| Sparse subsets of matrix entries | Local changes; record actual changed fraction |
| Whole scalar/vector function replacement | MOI support checking and assembly overhead |
| Reserved zero crossings | Fixed-pattern correctness |
| Genuine structural change | Exactly one appropriate rebuild |
| Large scaling/penalty/radius change | Warm-start and scaling robustness |

Use several representative horizons and dimensions, including many small SOCs and a large global SOC. Keep fixtures small enough for ordinary testing and use larger instances only in dedicated benchmarks. Report the exact cone layout, number of variables/rows, nnz(P/A/G), nnz(K), and nnz(L).

### B. End-to-end nonlinear SCP benchmark

Build a compact reproducible nonlinear optimal-control example that actually forms successive convex subproblems. A nondimensional orbital transfer with position/velocity states, thrust SOCs, dynamics linearization, a reference-centered trust region, endpoint constraints, and virtual-control penalties is appropriate. Use a supplied real SCP subproblem sequence when available; otherwise clearly label the example synthetic and document its dynamics and discretization.

Keep the example readable. Do not create a new trajectory-optimization framework merely to benchmark the solver. Verify derivatives/discretization on the small example and hold the outer SCP algorithm fixed when comparing solvers.

Include a realistic sequence of accepted steps, rejected/shrunk trust-region trials, and changed penalty weights. Verify the final nonlinear dynamics defect, endpoint accuracy, constraint satisfaction, objective, and termination condition. A fast convex subproblem solve that causes extra rejected outer iterations is not an automatic win.

Use both tests: replay isolates solver cost on identical problems; a live nonlinear loop measures the actual outcome even when numerical differences cause outer trajectories to diverge.

### C. Reference solvers and fair configuration

Compare at least:

1. Audited/current JuliaQOCO before the changes, with independent quality verification.
2. Improved JuliaQOCO, cold and reused, with appropriate warm-start/scaling modes.
3. C QOCO through its supported interface, separating wrapper/setup overhead from core solve time where possible.
4. Clarabel, including its supported fixed-pattern numerical-update route, not only fresh reconstruction.

Verify the installed versions and actual capabilities before writing adapters. Clarabel's documented data-update route has configuration restrictions, including presolve/chordal-decomposition settings; obey the version-specific contract. Do not claim a JuMP wrapper retains a native solver merely because that core supports updates. Show MOI-level and core-level comparisons separately when capabilities differ.

An OSQP comparison can be informative on genuinely quadratic-program-only cases, but do not compare a changed/approximated SOCP to the original conic problem. Do not make the benchmark depend on a commercial solver license.

Match numerical accuracy using independent checks, not only nominal tolerance settings. Include feasible accuracy targets such as `1e-4`, `1e-6`, and `1e-8` where supported and achieved. Keep failure counts and quality failures visible. Never loosen tolerances, change the optimization problem, or silently omit failed runs to claim a speedup.

### D. Measurements and report

Record:

- Julia/package/compiler versions, commit, CPU, operating system, Julia/BLAS threads, and solver settings;
- package load and first-solve latency separately from steady state;
- model construction, caller-side modification, native commit, scaling, initialization, NT construction, numeric factorization, triangular solves/refinement, and result extraction;
- full wall time for each update-and-solve and for the complete SCP run;
- median and p95 (or an explicitly justified distribution summary), allocations, and allocated bytes;
- IPM iterations, factorization count, refinement count, pivot regularizations, warm-start rejections/retries, rebuilds, and quality failures;
- original-unit solution quality and final nonlinear SCP quality;
- both the complete sequence cost and relevant per-iteration distributions.

Do not add inclusive phase timers together if they overlap. Make profiling overhead measurable and keep it off for headline timings. Exclude benchmark fixture generation and unrelated logging from kernel measurements, but include realistic model-update overhead in end-to-end measurements.

Publish machine-readable CSV/JSON results and a concise Markdown report. Show regressions as well as gains. A baseline instance that failed the quality gate cannot produce a meaningful accuracy-matched speedup ratio; label it explicitly.

## 11. Broaden tests and make integration behavior explicit

Expand the relevant MOI conformance coverage beyond the present small subset. Test supported primal/dual outputs, deletion/reindexing, all advertised modification routes, copying models, naming where advertised, result-index handling, and repeated solves. Keep exclusions narrow and explain them. Do not require infeasibility-certificate behavior that this algorithm does not implement.

Add focused unit/property/differential tests covering:

- original/scaled residual and objective identities;
- zero objectives, zero rows, empty/equality-only problems, and valid singular PSD Hessians;
- nonfinite/invalid data and settings with atomic rejection;
- SOC W/Winv identities and NT blocks over multiple dimensions and scales;
- line-search feasibility, including small positive steps below the old bisection resolution;
- fresh factorization versus cached numeric refactorization;
- KKT refinement against its explicitly defined target;
- updated models versus fresh reconstruction for all matrix/vector and MOI update forms;
- maximization whole-objective replacement versus coefficient modification;
- warm-start mode transitions and all four coordinates after rescaling;
- last-iteration assessment, restored-result consistency, and no-result failures;
- identical numerical behavior with profiling on and off;
- solver, symbolic pattern, and factor-storage reuse on fixed-pattern updates;
- a real nonlinear SCP smoke test with independent final-quality checks.

Use fixed random seeds and report failing cases reproducibly. Include small high-precision reference calculations where helpful, but do not promise arbitrary-precision solver support merely because types are parameterized. Make the supported numeric types explicit. Prioritize Float64 correctness and performance; only advertise Float32 or other types once their settings, factorization, and tolerances are properly exercised.

Add lightweight Ubuntu CI for the supported Julia 1 release and correctness/smoke tests if CI is absent. Keep expensive comparative timing benchmarks opt-in. Do not enforce brittle absolute runtime thresholds on shared CI runners; structure-reuse and measured allocation regressions can have more deterministic checks.

## 12. Precompilation, solve budgets, and diagnostics

The existing precompile workload exercises a tiny native orthant LP and a vector update. Measure first-use latency for the actual MOI/SOC/matrix-update routes and extend the small precompile workload where it demonstrably helps. Do not make JuMP a new runtime dependency just to precompile the wrapper. Avoid precompiling arbitrary horizons or exploding specializations for sparsity patterns.

Add a simple, documented solve-time budget through `MOI.TimeLimitSec` if feasible within the existing settings/status model. Use monotonic timing, check at useful boundaries, finalize a consistent available iterate, and account for any automatic retry under the same budget. State that a single native factorization call may not be preemptible; do not advertise a hard real-time deadline guarantee.

Make diagnostics distinguish structural rebuilds, numerical refactorizations, regularized pivots, retry attempts, and warm-start decisions. Correct the benchmark's iteration-count-based reuse claim. Define what native solve time includes and also expose/report the complete measured update-and-solve wall time where appropriate.

Keep the no-certificate limitation clear. Iteration limit, numerical failure, or a poorly scaled subproblem is not proof of infeasibility. An SCP controller may need to retry, change its trust region, or use another solver based on its own policy; the conic solver must provide truthful information rather than silently making that outer decision.

## 13. Documentation and expected deliverables

Update the README and focused developer documentation with:

- the supported mathematical form, objective/Hessian convention, cone ordering, and numeric types;
- a concise table of scaling and warm-start modes with accurate semantics;
- how to build one JuMP direct model, reserve structural support, update coefficients/RHSs/objective data, and query status safely;
- which operations trigger a structural rebuild and which only update numerical values;
- how residuals and objectives are computed in original units;
- the convexity-validation contract, invalid-data behavior, time-budget behavior, and infeasibility-certificate limitation;
- reproducible commands for tests, smoke examples, and benchmark suites;
- benchmark results and limitations, without unsupported performance claims.

From the repository root, establish the package environment with the normal local workflow, for example:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Adapt the actual benchmark commands to the implemented files and environments. Run every command you document where the environment permits. Do not label example commands as verified when they have not been executed.

Deliver the code changes, regression tests, maintained examples, source/algorithm notes where needed, reproducible benchmark harness, raw results, and a short results report. Keep benchmark output artifacts separate from runtime package code.

In the final implementation summary state exactly:

1. Which audited defects were reproduced and fixed, which were already fixed in the checkout, and which could not be reproduced.
2. Which optimizations were retained or rejected and why.
3. What tests and benchmarks actually ran, with the numerical outcomes and environment.
4. Accuracy-matched end-to-end improvements, regressions, and unresolved failures.
5. Any genuinely unexecuted verification or deferred experimental work.

Do not claim completion based only on the existing small test suite, a reused object identity, or a favorable no-change timing. Do not claim a speedup from a failed/less accurate solve. When an experimental representation does not earn its complexity, leave it out of production and record the evidence.

## Acceptance checklist

The mandatory P0/P1 implementation is complete when:

- [ ] Every success status is supported by independently checked original-unit quality.
- [ ] Zero-objective feasibility cases retain finite valid scaling and return a coherent result or an honest failure.
- [ ] Restored/current final vectors, objective, residuals, complementarity, and result availability agree.
- [ ] The final permitted IPM step is assessed before an iteration-limit decision.
- [ ] Whole-function, scalar, multirow, and bulk updates agree with a fresh construction.
- [ ] Concave maximization is converted correctly exactly once on every objective-update path.
- [ ] Invalid numerical updates cannot partially corrupt committed state.
- [ ] Warm-start modes behave as documented and all coordinates transform correctly under new scaling.
- [ ] Fixed-pattern updates retain symbolic analysis and factor storage without unnecessary rebuilds.
- [ ] Large whole-function support checks no longer use an all-pairs term scan.
- [ ] KKT/refinement behavior and nonfinite factorization failures are independently tested.
- [ ] Profiling, silence settings, interrupts, and result-index/status behavior are correct.
- [ ] Deterministic replay and a real nonlinear SCP example have independent quality gates.
- [ ] The comparative report separates cold, core, wrapper, repeated-update, and whole-SCP costs.
- [ ] No faster-but-wrong result is counted as a performance improvement.

Large-SOC expansion and secondary structural optimizations are additional measured improvements, not excuses to postpone these requirements.

## Evidence and primary references

The source links below are pinned to the audited revision. Inspect current equivalents before applying fixes.

| Source | Relevant evidence |
| --- | --- |
| [README](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/README.md) | Public API, existing reuse features, stated limitations |
| [kkt.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/kkt.jl) | Stopping criteria, KKT structure, refinement, IPM directions |
| [equilibration.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/equilibration.jl) | Scale conventions, zero-objective behavior, solution unscaling |
| [utils.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/utils.jl) | safe_div, weighted products, finiteness checks |
| [solver.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/solver.jl) | Setup validation, best-iterate logic, loop exits |
| [updates.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/updates.jl) | Numerical updates and warm-start transformations |
| [moi_wrapper.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/moi_wrapper.jl) | Full update path, objective sense, validation, statuses |
| [cones.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/cones.jl) | Compact NT multiplication, dense NT blocks, line search |
| [internal_qdldl.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/internal_qdldl.jl) | Existing cached numeric factorization and pivot handling |
| [JuliaQOCO.jl](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/src/JuliaQOCO.jl) | Exported API and precompile workload |
| [Native tests](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/test/native_tests.jl) | Existing numerical coverage |
| [MOI tests](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/test/moi_wrapper_tests.jl) | Public API policy, reuse and modification checks |
| [MOI conformance subset](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/test/moi_test_subset.jl) | Limited current conformance coverage and exclusions |
| [Repeated-solve benchmark](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/benchmark/jump_repeated_solves.jl) | Existing benchmark structure and measurement limitations |
| [Synthetic SCP-style example](https://github.com/jackyarndley/JuliaQOCO.jl/blob/c2725866335a51a54b10450726c22c05c4127dc9/examples/scp_jump_reuse.jl) | Current model and evolving update sequences |
| [QOCO paper, HTML](https://arxiv.org/html/2503.12658v4) | Intended Newton/refinement equations, stopping criteria, fixed-pattern custom-solver ideas |
| [Official QOCO documentation](https://qoco-org.github.io/qoco/index.html) | Upstream implementation and comparison context |
| [QOCO.jl](https://github.com/qoco-org/QOCO.jl) | C QOCO Julia interface; inspect supported update semantics/version |
| [Clarabel data-update documentation](https://clarabel.org/stable/user_guide_data_updating/) | Fixed-pattern update contract and configuration restrictions |
| [MOI expression/canonicalization documentation](https://jump.dev/MathOptInterface.jl/stable/tutorials/manipulating_expressions/) | Duplicate terms and canonical expression semantics |
| [MOI modification documentation](https://jump.dev/MathOptInterface.jl/stable/manual/modification/) | Whole-function replacement and coefficient-change contracts |

End of task. Implement and verify the prioritized improvements; use measurement to decide which speculative optimizations deserve to remain.
