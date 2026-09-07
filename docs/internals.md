# JuliaQOCO internals

Notes on the numerical conventions, for anyone changing the solver. The
authoritative statements are the comments at the top of `src/equilibration.jl`
and in `src/kkt.jl`; this document collects them and explains why they are
what they are.

## Problem form and data layout

```text
minimize    (1/2) x' P x + c' x
subject to  A x = b                     (p rows, dual y)
            h - G x in K                (m rows, dual z, slack s)
K = R^l_+  x  SOC(q[1])  x  SOC(q[2])  x  ...
```

`P` is stored upper-triangular in CSC. `data.At` and `data.Gt` are explicit
transposes with `AtoAt`/`GtoGt` index maps in both directions, so an update to
one nonzero reaches its transpose without a search. `data.Padded_idx` records
the diagonal entries that regularization inserted into `P`'s pattern and that
therefore do not correspond to any user coefficient.

## Scaling

With positive diagonal matrices `D` (variables), `E` (equality rows), `F` (cone
rows) and a positive objective scale `k`, the stored data is

```text
P_hat = k D P D     (before the static regularization shift)
c_hat = k D c
A_hat = E A D,  b_hat = E b
G_hat = F G D,  h_hat = F h
```

and the iterates relate to original units by

```text
x = D x_hat,   s = F^-1 s_hat,   y = E y_hat / k,   z = F z_hat / k.
```

Note the asymmetry: `s` carries `F^-1` where `z` carries `F`. This is what
makes complementarity convert as

```text
s' z = dot(s_hat, z_hat) / k
```

with **no** row weighting at all. Applying `F` to both, as a weighted dot
product would, introduces a spurious factor of `F^2`. The other identities the
stopping criteria rely on:

```text
x' P x   = dot(x_hat, P_hat x_hat) / k
c' x     = dot(c_hat, x_hat) / k
|c|_inf  = |D^-1 c_hat|_inf / k
|s|_inf  = |F^-1 s_hat|_inf
|b|_inf  = |E^-1 b_hat|_inf
|h|_inf  = |F^-1 h_hat|_inf
```

Three invariants must hold for any change to the equilibration:

1. **The objective scale stays finite and positive.** A problem with no
   objective at all, `P = 0` and `c = 0`, has nothing to equilibrate, and the
   neutral scale one is used. Dividing by a vanishing objective norm and
   accumulating the result over the Ruiz sweeps overflows `k` and destroys
   every unscaling that depends on it.
2. **Each second-order cone carries one common row scale.** Row scaling maps
   the cone onto itself only if the whole block is scaled by the same positive
   number. The common scale is the geometric mean of the per-row Ruiz scales,
   which is more representative than the head row alone. Cone membership is
   therefore preserved, which is why cone validity can be tested on the
   internal iterate.
3. **Every inverse scale is finite and positive.** `positive_inv` clamps, so a
   degenerate zero row cannot produce an infinite or zero inverse.

## Static regularization

`data.P` carries a `kkt_static_reg` shift on its diagonal, added *after*
equilibration. The mathematical Hessian is `data.P` minus that shift, and both
the reported objective and the original KKT residual subtract it. Reporting the
regularized objective as the requested objective would be wrong by an amount
proportional to `|x|^2`.

Because `k > 0` and `D > 0`, `P_hat` without the shift is a congruence
transform of `P` and has the same inertia. Convexity can therefore be verified
on the stored matrix without unscaling it.

## Numeric types and hard-coded constants

The solver is parameterized on its element type and is tested at `Float64` and
`Float32`. That imposes one discipline on every change: **no numerical
constant may be a bare `Float64` literal.**

The failure mode is not hypothetical. An absolute tolerance of `1e-7` is below
`eps(Float32)`, so a single-precision solve can never meet it. A static
regularization of `1e-8` added to a diagonal entry of order one is a no-op in
single precision. Both were bugs until the type-dependent defaults in
`settings.jl` replaced them; each rule is written as a floor at the tuned
`Float64` value plus a term proportional to `eps(T)`, so double precision keeps
exactly the values it was tuned with and lower precision gets something
achievable.

The subtler case is division. There are two genuinely different situations,
and they need different guards:

- A denominator that has **no business being small**, such as a total
  complementarity that has already collapsed. `checked_div` compares it
  against `safe_div_eps(T)` and reports `NaN` rather than fabricating a value.
- A denominator that is **expected to become small**, because the barrier
  drives it there: a slack, a dual, a cone determinant, a Nesterov-Todd scale.
  `bounded_ratio` computes the quotient whenever it is representable and
  saturates only on genuine overflow.

Conflating the two is what a single `safe_div` with a fixed epsilon did, and it
was wrong on its own terms even in double precision - it just could not be
reached there, because `1e-15` is below any dual the solver produces. In
single precision it is reached immediately: a dual of `1.6e-7` is an ordinary
iterate, and turning `w^2 = s/z` into `floatmax` for it collapses the step
length to `1e-25` and stalls the solve on a three-variable linear program.
`safe_div` no longer exists.

Scalar problem dimensions (`n`, `m`, `p`, `l`) are plain `Int` even when the
sparse index type is `Int32`. The index type governs the CSC arrays, where it
buys memory; propagating it into every offset and count buys nothing and makes
every helper signature a compatibility hazard.

## Three different KKT matrices

They are genuinely different objects and confusing them causes silent errors.

```text
Newton         K   = [ P     A'   G'    ]      the equation being solved
                     [ A     0    0     ]
                     [ G     0   -W'W   ]

factored     K_reg = [ P+eI   A'   G'         ]  what QDLDL factorizes
                     [ A     -eI   0          ]
                     [ G      0   -W'W - eI   ]

modified   the LDL factors after dynamic regularization has replaced any
           pivot whose sign disagreed with the requested inertia
```

`kkt_multiply!` implements the **Newton** matrix, with the static shift removed
from the primal block and no shift on the equality or cone blocks. Iterative
refinement uses the factorization of `K_reg` as a preconditioner and measures
its residual against `K`, so refinement converges to the solution of the
unregularized system. Adding the missing shifts to `kkt_multiply!` to make the
two matrices agree would change the algorithm, not fix it.

Dynamic regularization is an escalating retry, bounded to three attempts, and
it fires only on a `QDLDL.FactorizationFailure`. Interrupts and programming
errors propagate. The number of retries and the number of pivots that were
actually regularized are counted separately, because they answer different
questions.

## Sparse second-order-cone expansion

The Nesterov-Todd block of a second-order cone is dense. Storing it as an upper
triangle costs `q(q+1)/2` entries, which for one global trust region over a
whole trajectory is the dominant cost of the entire solve: measured on a
horizon-40 control fixture, rebuilding and refactorizing that block was 88% of
the solve time.

The block is not really dense, though. With `W = scale*(2 v v' - J)` and
`v'Jv = 1`, squaring gives

```text
W'W = scale^2 ( I + U M U' ),   U = [v  Jv],   M = [4 v'v  -2; -2  0].
```

`M` has determinant `-4`, so it has exactly one positive and one negative
eigenvalue. Splitting it along those eigenvectors turns the dense block into a
diagonal plus one positive and one negative rank-one term,

```text
W'W = scale^2 ( I + g g' - f f' ),
```

and because `g` and `f` are combinations of `v` and `Jv`, and `Jv` only flips
the tail sign, each is two scalars times the head and tail of `v`. Forming them
is `O(q)`.

Two auxiliary variables per cone then carry the rank-one terms:

```text
[ -scale^2 I    a      b   ]
[  a'           +1     0   ]
[  b'            0    -1   ]
```

Eliminating them reproduces `-W'W` exactly, provided `a a' / d1 = scale^2 g g'`
and `b b' / d2 = -scale^2 f f'`. This is an identity, not an approximation, and
the tests assert it to `1e-10` against the dense block.

Two details decide whether it works:

- **Quasidefiniteness.** The first auxiliary variable joins the primal block on
  the positive side of the partition and the second joins the dual block on the
  negative side, so the augmented matrix is still quasidefinite and the sign
  vector handed to the factorization is still meaningful.
- **The auxiliary pivots must be exactly ±1.** An auxiliary variable can be
  rescaled freely - coupling `kappa*g` with diagonal `kappa^2/scale^2` gives the
  same Schur complement for any `kappa` - but the static and dynamic
  regularizations are *absolute*. A pivot that drifts away from unit size has
  its rank-one term perturbed by a relative `eps/d`, which grows without bound
  as the iterate approaches the cone boundary, or is replaced outright by the
  dynamic regularization. Taking `kappa = scale` pins both pivots at `+1` and
  `-1` for the life of the solve. With the pivots left at `1/scale^2` the solve
  stalled at a duality gap of `1e-3`; pinned, it matches the dense path
  iteration for iteration.

The right-hand side is padded with zeros for the auxiliary rows and only the
leading part of the answer is kept, so `kkt_multiply!` and iterative refinement
continue to work entirely in the original space against the original Newton
operator.

`soc_expansion_threshold` chooses per cone. The crossover was measured on
problems with many cones of a single dimension: the dense block wins below
about dimension ten, and the expansion wins above it by a margin that keeps
growing.

## The shared accuracy metric

There is exactly one accuracy calculation, `solution_quality`, and every
decision uses it: regular stopping, inaccurate stopping, best-iterate ranking,
and the final status. It returns the largest of the three
residual-to-tolerance ratios,

```text
max( pres / (atol + rtol * p_ref),
     dres / (atol + rtol * d_ref),
     |gap| / (atol + rtol * g_ref) )
```

with the reference norms in original units, and `Inf` for an iterate that is
not finite or not in the cone. A value of at most one means every requested
tolerance is met. Because the same function is evaluated at the requested
tolerances and at the inaccurate ones, "solved" and "solved inaccurately" can
never disagree about which iterate is better.

`g_ref` uses `max(1, |primal objective|, |-0.5 x'Px - b'y - h'z|)`. The second
expression is a **scale**, not a certified dual bound: at an arbitrary
nonstationary iterate it is not a valid lower bound, and no dual objective is
reported from it.

## Finalization

`_finalize!` is the only place a result leaves the solver, and it always does
the same six things in the same order:

1. The most recently computed iterate has already been assessed by the caller,
   including the step taken on the final permitted iteration at the iteration
   limit.
2. The better of that iterate and the recorded best is chosen by the shared
   metric.
3. Objective, residuals, complementarity and cone validity are recomputed for
   whichever iterate was chosen.
4. That iterate is unscaled and published.
5. A termination reason and `result_available` are assigned from the
   recomputed metrics.
6. The warm-start cache is updated only for a usable result.

`_finalize_without_result!` handles the case where no iterate was ever computed
and publishes nothing.

`solution.iters` counts interior-point iterations performed;
`solution.result_iter` is the index of the iterate that was published. They are
separate fields because they answer separate questions, and a cold-start retry
makes them differ routinely.

## The line search

Every path returns `min(1, f * alpha_max)`. Starting from one rather than from
`f` lets an unrestricted direction take the full Newton step instead of being
damped for no reason.

For a second-order cone the boundary is the first positive root of

```text
alpha^2 (d0^2 - |d|^2) + 2 alpha (u0 d0 - <u,d>) + (u0^2 - |u|^2) = 0.
```

Both blocks are normalized by their largest entry before the coefficients are
formed, which keeps all three of order one whatever the data magnitude, and the
two determinants are evaluated as `(head - tail)(head + tail)` rather than as a
difference of squares, because near the boundary the latter loses essentially
all of its significant digits.

When the analytical solution is not trustworthy - a start that is not usably
interior, or a concave quadratic with no root - the caller falls back to
`safeguarded_linesearch!`, which tries the full step, backtracks geometrically
up to thirty times, and then refines the bracket. It returns zero only when no
positive step above roughly `1e-9` stays inside the cone. The five-step
bisection this replaces could only ever return zero or a multiple of `1/32`,
and so reported "no step" for any feasible step below `1/32`.

## Profiled and unprofiled runs

The interior-point loop is written once, with `PROFILE` as a compile-time
`Val` parameter, so the instrumentation vanishes from the fast path and the
numerical work cannot drift between the two. `test/numerics_tests.jl` asserts
bit-for-bit equality of the results.

## Warm-start coordinate conversion

`_warmstart_to_scaled!` and `_warmstart_to_original!` are exact inverses of
each other and of `unscaled_solution!`, and each converts **all four**
components together. Converting them in separate branches is what previously
allowed `z` to be left behind in an old scaling after a recomputed
equilibration.

Reconstructing the slack from `x` is a separate policy decision, applied after
the conversion, not part of it.

## Structural optimizations that were assessed and not pursued

Both were considered against measurements rather than intuition, and neither
earned its complexity once the second-order-cone expansion had removed the
dominant cost.

**A direct solve path for equality-only quadratic programs.** An
equality-constrained quadratic program with no cone rows already terminates in
*zero* interior-point iterations: the initialization solve is the Karush-Kuhn-Tucker
solve, and the stopping check accepts it immediately. Measured at
`n = 200`, `p = 80`, a cold solve takes 0.054 ms and reports a primal residual
of `1e-15`. There is no repeated work for a special path to remove.

**Eliminating one-variable orthant rows.** A simple bound contributes a row
and a column to the Karush-Kuhn-Tucker matrix, and eliminating its dual
direction would fold `G' D^-1 G` into the primal diagonal only. That is a real
structural reduction, but after the cone expansion the whole solve of the
medium control fixture is 0.41 ms, down from 4.31 ms, and the orthant rows are
a minority of what remains. The change needs its own scaling path, right-hand
side transformation and direction recovery, so it is not worth the risk at
that scale. Revisit it only with a profile that shows the orthant block
dominating.

## Update paths

Every entry point in `src/updates.jl` validates before it mutates. The single
exception is the convexity test, which needs the new Hessian to exist: that
path snapshots the previous values and restores them, together with the factor
values, if the test fails.

`update_data!` is the one transaction entry point. It takes all six data
fields, so that a recomputed scaling sees the complete new problem.
`update_matrix_data!` is a thin wrapper for matrix-only transactions.

## MathOptInterface layer

The editable model is a `UniversalFallback`; numerical assembly is cached
separately. Two consequences worth knowing:

- The fallback claims to support every attribute, so any attribute this solver
  does not actually produce must be declined **by name**. Forwarding
  `MOI.supports` to the fallback would advertise a basis and a dual objective
  that are never computed.
- Objective-sense conversion is applied **exactly once**, by whichever caller
  queues a coefficient. The stored target multipliers are one for quadratic
  entries. Folding the sense into the target as well applies it twice on whole
  replacement, which turns a concave maximization into a nonconvex
  minimization; there is a regression test for exactly that.

Functions are canonicalized at the boundary, so duplicate terms accumulate
instead of overwriting one another. That is separate from the rejection of
duplicate row entries in native CSC storage, which stays a hard error.

Support checks probe the cached coordinate map rather than comparing every old
term against every new one, so a whole-function replacement costs time
proportional to the number of terms rather than to its square.
