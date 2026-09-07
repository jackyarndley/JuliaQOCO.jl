# Numerical regression tests, all validated against the independent oracle in
# `oracle.jl` rather than against the solver's own residual code.

using Test
using Random
using LinearAlgebra
using SparseArrays
using JuliaQOCO

include("oracle.jl")
using .Oracle

const Q = JuliaQOCO

quiet(; kwargs...) = Q.Settings{Float64}(; verbose = false, kwargs...)

# Build a solver and the matching oracle problem from one description, so the
# two can never drift apart.
function build(P, c, A, b, G, h, l, q; kwargs...)
    settings = quiet(; kwargs...)
    solver = Q.CoreSolver(P, c, A, b, G, h, l, q; settings = settings)
    problem = OracleProblem(P, c, A, b, G, h, l, q)
    return solver, problem
end

# --- fixtures ---------------------------------------------------------------

# Equality-constrained QP with a coupled positive definite Hessian and bounds.
function coupled_qp()
    P = sparse([1, 1, 2, 3], [1, 2, 2, 3], [2.0, 0.5, 3.0, 1.0], 3, 3)
    c = [-1.0, -2.0, 0.5]
    A = sparse([1, 1, 1], [1, 2, 3], [1.0, 1.0, 1.0], 1, 3)
    b = [1.0]
    G = sparse(1:3, 1:3, fill(-1.0, 3), 3, 3)
    h = zeros(3)
    return P, c, A, b, G, h, 3, Int[]
end

# Singular positive semidefinite Hessian: the third variable has no curvature.
function singular_psd_qp()
    P = sparse([1, 2], [1, 2], [2.0, 1.0], 3, 3)
    c = [-1.0, -1.0, 0.25]
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 3)
    b = [1.0]
    G = sparse(1:3, 1:3, fill(-1.0, 3), 3, 3)
    h = zeros(3)
    return P, c, A, b, G, h, 3, Int[]
end

# Mixed problem: orthant bounds, one equality, and one second-order cone.
# Every variable carries curvature, so the problem stays bounded and its
# optimum stays unique under the objective perturbations used below.
function mixed_socp()
    n = 4
    P = sparse([1, 1, 2, 3, 4], [1, 2, 2, 3, 4], [1.0, 0.3, 1.0, 0.5, 0.5], n, n)
    c = [0.1, -0.4, 0.05, 0.07]
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, n)
    b = [0.6]
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    # Two orthant rows: x3 >= 0 and x4 >= 0.
    push!(rows, 1); push!(cols, 3); push!(vals, -1.0)
    push!(rows, 2); push!(cols, 4); push!(vals, -1.0)
    # SOC of dimension 3 on (1.5, x1, x2).
    push!(rows, 4); push!(cols, 1); push!(vals, -1.0)
    push!(rows, 5); push!(cols, 2); push!(vals, -1.0)
    G = sparse(rows, cols, vals, 5, n)
    h = [0.0, 0.0, 1.5, 0.0, 0.0]
    return P, c, A, b, G, h, 2, [3]
end

# Feasibility problem: no objective at all.
function zero_objective_feasibility(scale::Float64)
    n = 3
    P = spzeros(n, n)
    c = zeros(n)
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, n)
    b = [1.0]
    G = sparse(1:n, 1:n, fill(-scale, n), n, n)
    h = zeros(n)
    return P, c, A, b, G, h, n, Int[]
end

# Equality-only QP with no cone rows at all.
function equality_only_qp()
    P = sparse([1, 2], [1, 2], [2.0, 4.0], 2, 2)
    c = [-1.0, 1.0]
    A = sparse([1, 1], [1, 2], [1.0, 2.0], 1, 2)
    b = [0.5]
    return P, c, A, b, nothing, nothing, 0, Int[]
end

# --- scaling algebra and stopping criteria ---------------------------------

@testset "Original-unit stopping criteria" begin
    for (name, fixture) in (
        ("coupled QP", coupled_qp()),
        ("singular PSD QP", singular_psd_qp()),
        ("mixed SOCP", mixed_socp()),
        ("equality-only QP", equality_only_qp()),
    )
        solver, problem = build(fixture...)
        Q._solve!(solver)
        @test solver.solution.status == Q.QOCO_SOLVED
        @test solver.solution.result_available
        check_result(problem, solver.solution; label = name)
    end
end

@testset "Reported metrics survive nonunit scaling" begin
    # A large uniform cone-row scale is exactly the case that exposed the
    # squared row-scaling factor in the complementarity calculation: the
    # reported gap came out F^2 times too small.
    for cone_scale in (1.0, 1e-4, 1e4)
        P, c, A, b, G, h, l, q = zero_objective_feasibility(1.0)
        G = G .* cone_scale
        solver, problem = build(P, c, A, b, G, h, l, q)
        Q._solve!(solver)
        @test solver.scaling.k > 0 && isfinite(solver.scaling.k)
        @test all(isfinite, solver.scaling.Fruiz)
        @test all(>(0), solver.scaling.Fruiz)
        report = check_result(problem, solver.solution; label = "cone scale $cone_scale")
        # The complementarity the solver reports is the true one, not one
        # rescaled by a stray factor of the row scaling.
        @test isapprox(solver.solution.gap, report.complementarity; atol = 1e-12, rtol = 1e-8)
    end
end

@testset "Scaled transformations of one problem agree" begin
    P, c, A, b, G, h, l, q = coupled_qp()
    reference, reference_problem = build(P, c, A, b, G, h, l, q)
    Q._solve!(reference)
    check_result(reference_problem, reference.solution; label = "reference")

    # Multiply the equality row by 1000 and scale the objective by 1e-3. The
    # transformed problem has the same primal optimum.
    A2 = A .* 1000.0
    b2 = b .* 1000.0
    P2 = P .* 1e-3
    c2 = c .* 1e-3
    solver, problem = build(P2, c2, A2, b2, G, h, l, q)
    Q._solve!(solver)
    @test solver.solution.status == Q.QOCO_SOLVED
    check_result(problem, solver.solution; label = "transformed")
    # The transformed objective is a thousand times smaller, so the two runs
    # stop at different absolute distances from the active bound. Compare the
    # solution quality and the (rescaled) objective, not the last digits of a
    # variable sitting on a boundary.
    @test isapprox(solver.solution.x, reference.solution.x; atol = 1e-4)
    @test isapprox(solver.solution.obj * 1e3, reference.solution.obj; rtol = 1e-5)
end

@testset "Zero objective keeps a finite scaling" begin
    for scale in (1.0, 1e-8, 1e8)
        P, c, A, b, G, h, l, q = zero_objective_feasibility(scale)
        solver, problem = build(P, c, A, b, G, h, l, q)
        @test isfinite(solver.scaling.k)
        @test solver.scaling.k > 0
        @test isfinite(solver.scaling.kinv)
        @test all(isfinite, solver.scaling.Dinvruiz)
        Q._solve!(solver)
        @test solver.solution.status in (Q.QOCO_SOLVED, Q.QOCO_SOLVED_INACCURATE)
        check_result(problem, solver.solution; atol = 1e-5, rtol = 1e-5, label = "zero objective $scale")
    end
end

@testset "Success is never implied by a degenerate metric" begin
    solver, _ = build(coupled_qp()...)
    Q._solve!(solver)
    # Force a nonfinite reported quantity and confirm the shared metric
    # refuses to call it a success.
    solver.solution.pres = NaN
    @test !isfinite(Q.solution_quality(solver, 1e-7, 1e-7))
    solver.solution.pres = 0.0
    solver.solution.gap = Inf
    @test !isfinite(Q.solution_quality(solver, 1e-7, 1e-7))
    solver.solution.gap = 0.0
    solver.solution.cone_valid = false
    @test !isfinite(Q.solution_quality(solver, 1e-7, 1e-7))
end

# --- finalization -----------------------------------------------------------

@testset "Finalization publishes one coherent result" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    for iterations in (1, 2, 3, 200)
        solver, problem = build(P, c, A, b, G, h, l, q; max_iters = iterations)
        Q._solve!(solver)
        sol = solver.solution
        @test sol.iters <= iterations
        @test all(isfinite, sol.x)
        if sol.result_available
            # Reported metrics must describe the returned vectors even when
            # the returned vectors came from an earlier iterate.
            check_result(problem, sol; atol = 1.0, rtol = 1.0, label = "max_iters=$iterations")
            @test sol.result_iter <= sol.iters
        end
    end
end

@testset "Restored best iterate stays consistent" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, problem = build(P, c, A, b, G, h, l, q; max_iters = 30)
    Q._solve!(solver)
    @test solver.solution.result_available
    # Force the final iterate to be worse than a recorded best by rerunning
    # with a deliberately tiny iteration budget, then check that whichever
    # iterate is published is the one described by the metrics.
    solver2, problem2 = build(P, c, A, b, G, h, l, q; max_iters = 2)
    Q._solve!(solver2)
    if solver2.solution.result_available
        check_result(problem2, solver2.solution; atol = 1.0, rtol = 1.0, label = "restored")
    end
end

@testset "Unusable data produces no available result" begin
    solver, _ = build(coupled_qp()...)
    # Nonfinite data poisons the initialization solve. Whatever the loop does
    # with it, no result may be advertised as available.
    solver.data.c[1] = NaN
    Q._solve!(solver)
    @test solver.solution.status == Q.QOCO_NUMERICAL_ERROR
    @test !solver.solution.result_available
    @test solver.solution.iters <= 1
    @test !solver.warmstart.active

    # A factorization that fails outright before any iterate exists must
    # publish nothing at all, not a stale or all-zero vector.
    solver2, _ = build(coupled_qp()...)
    Q._solve!(solver2)
    @test solver2.solution.result_available
    fill!(solver2.linsys.factor.workspace.triuA.nzval, NaN)
    solver2.warmstart.active = false
    Q._solve!(solver2)
    @test solver2.solution.status == Q.QOCO_NUMERICAL_ERROR
    @test !solver2.solution.result_available
    @test all(iszero, solver2.solution.x)
end

@testset "Profiling does not change the numbers" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    plain, _ = build(P, c, A, b, G, h, l, q)
    Q._solve!(plain)
    profiled, _ = build(P, c, A, b, G, h, l, q; profile = true)
    Q._solve!(profiled)
    @test plain.solution.status == profiled.solution.status
    @test plain.solution.iters == profiled.solution.iters
    @test plain.solution.x == profiled.solution.x
    @test plain.solution.s == profiled.solution.s
    @test plain.solution.y == profiled.solution.y
    @test plain.solution.z == profiled.solution.z
    @test plain.solution.obj == profiled.solution.obj
    @test plain.solution.pres == profiled.solution.pres
    @test plain.solution.dres == profiled.solution.dres
    @test plain.solution.gap == profiled.solution.gap
    @test profiled.solution.profile.linsys_solves > 0
end

@testset "Time budget terminates honestly" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, _ = build(P, c, A, b, G, h, l, q; time_limit_sec = 1e-9, max_iters = 200)
    Q._solve!(solver)
    @test solver.solution.status in
          (Q.QOCO_TIME_LIMIT, Q.QOCO_SOLVED, Q.QOCO_SOLVED_INACCURATE)
    @test solver.solution.iters <= 2
end

# --- cone kernels and the line search --------------------------------------

@testset "SOC determinant is accurate near the boundary" begin
    # A point a relative 1e-12 inside the boundary: the naive difference of
    # squares loses the value entirely, the factored form keeps it.
    tail = [3.0, 4.0]
    head = 5.0 * (1 + 1e-12)
    u = [head; tail]
    expected = head^2 - 25.0
    @test isapprox(Q.soc_residual2(u, 1, 3), expected; rtol = 1e-6)
    # Overflow-safe tail norm.
    big = [1e200, 1e200]
    @test isapprox(Q.soc_tail_norm([0.0; big], 1, 3), sqrt(2) * 1e200; rtol = 1e-12)
end

@testset "Line search finds tiny positive steps" begin
    # An orthant point whose feasible step is far below the resolution of the
    # five-iteration bisection the fallback replaces.
    solver, _ = build(
        spzeros(1, 1), [0.0], nothing, nothing,
        sparse([1], [1], [-1.0], 3, 1), [1.0, 1.0, 1.0], 0, [3],
    )
    u = [1.0, 0.0, 0.0]
    # Direction that leaves the cone after about 1e-6.
    Du = [-1.0, 1e6, 0.0]
    step = Q.linesearch!(solver, u, Du, 0.99)
    @test step > 0
    @test step < 1e-5
    trial = u .+ step .* Du
    @test Q.cone_residual(trial, 0, [3]) < 0

    # A direction that never leaves the cone must be allowed a unit step.
    @test Q.linesearch!(solver, u, [1.0, 0.0, 0.0], 0.99) == 1.0
    # A zero direction is unrestricted.
    @test Q.linesearch!(solver, u, [0.0, 0.0, 0.0], 0.99) == 1.0
end

@testset "Line search agrees with a reference boundary" begin
    rng = MersenneTwister(20260907)
    solver, _ = build(
        spzeros(1, 1), [0.0], nothing, nothing,
        sparse([1], [1], [-1.0], 4, 1), ones(4), 0, [4],
    )
    for _ in 1:200
        tail = randn(rng, 3)
        u = [norm(tail) + 0.5 + rand(rng); tail]
        Du = randn(rng, 4)
        step = Q.linesearch!(solver, u, Du, 1.0)
        @test 0 <= step <= 1
        if step > 0
            @test Q.cone_residual(u .+ (step * (1 - 1e-9)) .* Du, 0, [4]) <= 1e-9
        end
        if step < 1
            # Just past the returned step the point must be outside.
            beyond = u .+ (step * 1.001 + 1e-12) .* Du
            @test Q.cone_residual(beyond, 0, [4]) > -1e-6
        end
    end
end

# --- factorization ----------------------------------------------------------

@testset "Nonfinite pivots are detected" begin
    K = sparse([1, 1, 2], [1, 2, 2], [1.0, 1.0, -1.0], 2, 2)
    factor = Q.QDLDL.qdldl(K; Dsigns = [1, -1])
    Q.QDLDL.update_values!(factor, [1], [NaN])
    @test_throws Q.QDLDL.FactorizationFailure Q.QDLDL.refactor!(factor)
end

@testset "Cached refactorization matches a fresh factorization" begin
    rng = MersenneTwister(11)
    n = 12
    # A genuinely quasidefinite matrix: a positive definite leading block, a
    # negative definite trailing block, and a sparse coupling between them.
    # The shifts are chosen from the actual norms so that both blocks are
    # strictly diagonally dominant and the inertia is known in advance.
    half = n ÷ 2
    top = sprandn(rng, half, half, 0.4)
    bottom = sprandn(rng, half, half, 0.4)
    coupling = sprandn(rng, half, half, 0.3)
    upper = top + transpose(top)
    lower = bottom + transpose(bottom)
    upper += (norm(Matrix(upper), Inf) + 1.0) * I
    lower -= (norm(Matrix(lower), Inf) + 1.0) * I
    pattern = triu([
        upper                 coupling
        spzeros(half, half)   lower
    ])
    signs = vcat(ones(Int, half), -ones(Int, n - half))
    inertia = eigvals(Symmetric(Matrix(pattern), :U))
    @test count(>(0), inertia) == half
    factor = Q.QDLDL.qdldl(SparseMatrixCSC(pattern); Dsigns = signs)
    rhs = randn(rng, n)
    # Pick an off-diagonal slot so that one trial can drive a coefficient
    # through zero without making the matrix itself singular.
    columns = Q.entry_columns(pattern)
    off_diagonal = findfirst(k -> pattern.rowval[k] != columns[k], 1:nnz(pattern))
    @test off_diagonal !== nothing
    for trial in 1:5
        values = copy(pattern.nzval)
        values[off_diagonal] =
            trial == 3 ? 0.0 : values[off_diagonal] * (1 + 0.1 * trial)
        M = SparseMatrixCSC(n, n, copy(pattern.colptr), copy(pattern.rowval), values)
        Q.QDLDL.update_values!(factor, collect(1:nnz(pattern)), values)
        Q.QDLDL.refactor!(factor)
        cached = Q.QDLDL.solve(factor, copy(rhs))
        fresh = Q.QDLDL.qdldl(M; Dsigns = signs)
        direct = Q.QDLDL.solve(fresh, copy(rhs))
        @test isapprox(cached, direct; rtol = 1e-8, atol = 1e-10)
        @test isapprox(Symmetric(Matrix(M), :U) * cached, rhs; rtol = 1e-6, atol = 1e-8)
    end
end

@testset "KKT product matches its documented target" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, _ = build(P, c, A, b, G, h, l, q)
    Q.initialize_ipm!(solver)
    Q.compute_kkt_residual!(solver)
    Q.compute_nt_scaling!(solver)
    data = solver.data
    work = solver.work
    n, p, m = Int(data.n), Int(data.p), Int(data.m)
    reg = solver.settings.kkt_static_reg

    # Assemble the original Newton matrix explicitly: the mathematical Hessian
    # (stored P minus the static shift), the constraint blocks, and -W'W.
    Pmath = Matrix(Symmetric(Matrix(data.P), :U)) - reg * I
    W = zeros(m, m)
    for j in 1:m
        e = zeros(m)
        e[j] = 1.0
        Q.nt_multiply_W!(view(W, :, j), e, data, work)
    end
    K = [
        Pmath                Matrix(data.At)   Matrix(data.Gt)
        Matrix(data.A)       zeros(p, p)       zeros(p, m)
        Matrix(data.G)       zeros(m, p)       -transpose(W) * W
    ]
    rng = MersenneTwister(3)
    for _ in 1:10
        v = randn(rng, n + p + m)
        out = similar(v)
        Q.kkt_multiply!(out, v, data, work, reg)
        @test isapprox(out, K * v; rtol = 1e-9, atol = 1e-10)
    end
end

# --- sparse second-order-cone expansion -------------------------------------

# A problem with one cone big enough to be expanded, plus orthant rows.
function big_cone_socp(qdim::Int; n::Int = 10)
    rng = MersenneTwister(4)
    P = sparse(1:n, 1:n, rand(rng, n) .+ 0.5, n, n)
    c = randn(rng, n)
    A = sparse(ones(Int, n), 1:n, ones(n), 1, n)
    b = [1.0]
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for i in 1:n
        push!(rows, i); push!(cols, i); push!(vals, -1.0)
    end
    h = zeros(n)
    push!(h, 3.0)
    for k in 1:(qdim - 1)
        push!(rows, n + 1 + k); push!(cols, mod1(k, n)); push!(vals, -1.0); push!(h, 0.0)
    end
    G = sparse(rows, cols, vals, n + qdim, n)
    return P, c, A, b, G, h, n, [qdim]
end

@testset "Expanded cone block reproduces the dense one" begin
    rng = MersenneTwister(31)
    for qdim in (4, 9, 30)
        P, c, A, b, G, h, l, q = big_cone_socp(qdim)
        dense, _ = build(P, c, A, b, G, h, l, q; soc_expansion_threshold = typemax(Int))
        expanded, _ = build(P, c, A, b, G, h, l, q; soc_expansion_threshold = 2)
        @test !any(dense.work.soc_expanded)
        @test all(expanded.work.soc_expanded)
        @test isempty(dense.work.soc_aux)
        @test length(expanded.work.soc_aux) == 2 * qdim + 2
        # The expansion trades a quadratic triangular block for a linear one.
        @test length(expanded.work.WtW) < length(dense.work.WtW)

        for _ in 1:5
            tail = randn(rng, qdim - 1)
            head = norm(tail) + 0.5 + rand(rng)
            tail2 = randn(rng, qdim - 1)
            head2 = norm(tail2) + 0.5 + rand(rng)
            s = vcat(0.4 .+ rand(rng, l), [head], tail)
            z = vcat(0.4 .+ rand(rng, l), [head2], tail2)
            for solver in (dense, expanded)
                copyto!(solver.work.s, s)
                copyto!(solver.work.z, z)
                Q.compute_nt_scaling!(solver)
            end

            # Reconstruct the cone block from each representation and compare.
            offset = Int(dense.work.Wtri_offsets[1])
            reference = zeros(qdim, qdim)
            for j in 1:qdim, k in 1:j
                value = dense.work.WtW[offset + (j * (j - 1)) ÷ 2 + k - 1]
                reference[k, j] = value
                reference[j, k] = value
            end
            aux = Int(expanded.work.soc_aux_offsets[1])
            g = expanded.work.soc_aux[aux:(aux + qdim - 1)]
            d1 = expanded.work.soc_aux[aux + qdim]
            f = expanded.work.soc_aux[(aux + qdim + 1):(aux + 2 * qdim)]
            d2 = expanded.work.soc_aux[aux + 2 * qdim + 1]
            scale2 = expanded.work.WtW[Int(expanded.work.Wtri_offsets[1])]
            # Eliminating the two auxiliary variables must give back exactly
            # the dense block.
            reconstructed = scale2 * Matrix(I, qdim, qdim) + g * g' / d1 + f * f' / d2
            @test isapprox(reconstructed, reference; rtol = 1e-10, atol = 1e-12)
            # The auxiliary pivots are pinned to plus and minus one, which is
            # what keeps the absolute regularization from perturbing them.
            @test d1 == 1.0
            @test d2 == -1.0
        end
    end
end

@testset "Expanded cones solve identically to dense ones" begin
    for qdim in (4, 9, 30, 60)
        P, c, A, b, G, h, l, q = big_cone_socp(qdim)
        dense, problem = build(P, c, A, b, G, h, l, q; soc_expansion_threshold = typemax(Int))
        expanded, _ = build(P, c, A, b, G, h, l, q; soc_expansion_threshold = 2)
        Q._solve!(dense)
        Q._solve!(expanded)
        @test dense.solution.status == Q.QOCO_SOLVED
        @test expanded.solution.status == Q.QOCO_SOLVED
        check_result(problem, expanded.solution; label = "expanded q=$qdim")
        @test dense.solution.iters == expanded.solution.iters
        @test isapprox(dense.solution.obj, expanded.solution.obj; rtol = 1e-8)
        @test isapprox(dense.solution.x, expanded.solution.x; atol = 1e-7)
        @test isapprox(dense.solution.z, expanded.solution.z; atol = 1e-7)
    end
end

@testset "Expanded cones survive updates and warm starts" begin
    qdim = 40
    P, c, A, b, G, h, l, q = big_cone_socp(qdim)
    solver, _ = build(P, c, A, b, G, h, l, q; soc_expansion_threshold = 8)
    Q._solve!(solver)
    factor = solver.linsys.factor
    for trial in 1:4
        newh = copy(h)
        newh[l + 1] = 3.0 + 0.3 * trial
        newc = c .* (1 + 0.05 * trial)
        Q.update_data!(solver; c = newc, h = newh)
        Q._solve!(solver)
        fresh, problem = build(P, newc, A, b, G, newh, l, q; soc_expansion_threshold = 8)
        Q._solve!(fresh)
        @test solver.linsys.factor === factor
        @test solver.solution.status == Q.QOCO_SOLVED
        check_result(problem, solver.solution; label = "expanded update $trial")
        @test isapprox(solver.solution.obj, fresh.solution.obj; atol = 1e-6, rtol = 1e-5)
    end
end

# --- updates ----------------------------------------------------------------

@testset "Updated models match fresh construction" begin
    rng = MersenneTwister(7)
    P, c, A, b, G, h, l, q = mixed_socp()
    for mode in (:once, :recompute)
        solver, _ = build(P, c, A, b, G, h, l, q; scaling_mode = mode)
        Q._solve!(solver)
        for trial in 1:4
            newc = c .+ 0.1 * trial .* randn(rng, length(c))
            newb = b .+ 0.05 * randn(rng, length(b))
            newh = copy(h)
            newh[3] = 1.5 + 0.2 * trial
            newGx = G.nzval .* (1 .+ 0.05 * trial)
            newG = SparseMatrixCSC(size(G, 1), size(G, 2), copy(G.colptr), copy(G.rowval), newGx)
            Q.update_data!(solver; Gx = newGx, c = newc, b = newb, h = newh)
            Q._solve!(solver)
            fresh, problem = build(P, newc, A, newb, newG, newh, l, q; scaling_mode = mode)
            Q._solve!(fresh)
            @test solver.solution.status == Q.QOCO_SOLVED
            check_result(problem, solver.solution; label = "update $mode trial $trial")
            # Matched accuracy: both runs stopped at the same requested
            # tolerance, so they agree to that tolerance and no further.
            @test isapprox(solver.solution.obj, fresh.solution.obj; atol = 1e-6, rtol = 1e-5)
            @test isapprox(solver.solution.x, fresh.solution.x; atol = 1e-5)
        end
    end
end

@testset "Invalid updates leave committed state intact" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, problem = build(P, c, A, b, G, h, l, q)
    Q._solve!(solver)
    reference_c = copy(solver.data.c)
    reference_G = copy(solver.data.G.nzval)
    reference_P = copy(solver.data.P.nzval)

    @test_throws ArgumentError Q.update_vector_data!(solver; c = fill(NaN, length(c)))
    @test solver.data.c == reference_c
    @test_throws ArgumentError Q.update_c_entries!(solver, [1], [Inf])
    @test solver.data.c == reference_c
    @test_throws ArgumentError Q.update_c_entries!(solver, [99], [1.0])
    @test solver.data.c == reference_c
    @test_throws ArgumentError Q.update_G_entries!(solver, [1], [NaN])
    @test solver.data.G.nzval == reference_G

    # A nonconvex Hessian update is rejected and rolled back.
    @test_throws ArgumentError Q.update_P_entries!(solver, [1], [-5.0])
    @test solver.data.P.nzval == reference_P
    Q._solve!(solver)
    @test solver.solution.status == Q.QOCO_SOLVED
    check_result(problem, solver.solution; label = "after rejected updates")
end

@testset "Convexity policy covers every route" begin
    n = 3
    diagonal = sparse(1:n, 1:n, [1.0, 2.0, 3.0], n, n)
    indefinite_diagonal = sparse(1:n, 1:n, [1.0, -2.0, 3.0], n, n)
    coupled = sparse([1, 1, 2], [1, 2, 2], [1.0, 5.0, 1.0], n, n)
    c = zeros(n)
    G = sparse(1:n, 1:n, fill(-1.0, n), n, n)
    h = ones(n)

    @test Q.is_diagonal_pattern(diagonal)
    @test !Q.is_diagonal_pattern(coupled)
    @test Q.is_positive_semidefinite(diagonal, 0.0, 512)
    @test !Q.is_positive_semidefinite(indefinite_diagonal, 0.0, 512)
    # Same verdict whether the dense or the shifted-Cholesky branch is used.
    @test !Q.is_positive_semidefinite(coupled, 0.0, 512)
    @test !Q.is_positive_semidefinite(coupled, 0.0, 0)
    psd_coupled = sparse([1, 1, 2], [1, 2, 2], [2.0, 1.0, 2.0], n, n)
    @test Q.is_positive_semidefinite(psd_coupled, 0.0, 512)
    @test Q.is_positive_semidefinite(psd_coupled, 0.0, 0)
    # A singular positive semidefinite matrix must be accepted, not rejected
    # as merely "not positive definite".
    singular = sparse([1, 1, 2], [1, 2, 2], [1.0, 1.0, 1.0], n, n)
    @test Q.is_positive_semidefinite(singular, 0.0, 512)
    @test Q.is_positive_semidefinite(singular, 0.0, 0)

    @test_throws ArgumentError Q.CoreSolver(
        indefinite_diagonal, c, nothing, nothing, G, h, n, Int[]; settings = quiet(),
    )
    # The caller-guarantee escape hatch skips the test rather than pretending
    # the matrix was verified.
    solver = Q.CoreSolver(
        indefinite_diagonal, c, nothing, nothing, G, h, n, Int[];
        settings = quiet(convexity_check = :none),
    )
    @test solver isa Q.CoreSolver
end

@testset "Structural zeros stay inside the allocated pattern" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, _ = build(P, c, A, b, G, h, l, q)
    Q._solve!(solver)
    factor = solver.linsys.factor
    original = copy(solver.data.G.nzval)
    Q.update_G_entries!(solver, [1], [0.0])
    Q._solve!(solver)
    @test solver.linsys.factor === factor
    Q.update_G_entries!(solver, [1], [-1.0])
    Q._solve!(solver)
    @test solver.linsys.factor === factor
    @test isapprox(solver.data.G.nzval, original; rtol = 1e-12)
end

@testset "Failure is never dressed up as success" begin
    # Infeasible: the equality forces a norm of at least 0.6/sqrt(2), which the
    # cone radius forbids. The solver has no infeasibility certificate, so the
    # only correct behaviour is an honest failure, never OPTIMAL.
    P, c, A, b, G, h, l, q = mixed_socp()
    infeasible_h = copy(h)
    infeasible_h[3] = 1e-3
    solver, _ = build(P, c, A, b, G, infeasible_h, l, q)
    Q._solve!(solver)
    @test solver.solution.status != Q.QOCO_SOLVED
    @test !(solver.solution.result_available && solver.solution.quality <= 1)

    # Unbounded below: a variable with no curvature and a negative cost, free
    # to grow along the nonnegative orthant.
    unbounded_c = copy(c)
    unbounded_c[3] = -1.0
    unbounded_P = sparse([1, 2], [1, 2], [1.0, 1.0], 4, 4)
    solver = Q.CoreSolver(
        unbounded_P, unbounded_c, A, b, G, h, l, q; settings = quiet(max_iters = 60),
    )
    Q._solve!(solver)
    @test solver.solution.status != Q.QOCO_SOLVED

    # A nonconvex Hessian slipped past the caller guarantee must not come back
    # as a certified optimum just because the regularized factorization
    # happened to succeed.
    nonconvex = sparse([1, 2], [1, 2], [1.0, -1.0], 2, 2)
    solver = Q.CoreSolver(
        nonconvex, [0.0, 0.0], nothing, nothing,
        sparse([1, 2, 3, 4], [1, 2, 1, 2], [-1.0, -1.0, 1.0, 1.0], 4, 2),
        [1.0, 1.0, 1.0, 1.0], 4, Int[];
        settings = quiet(convexity_check = :none, max_iters = 60),
    )
    Q._solve!(solver)
    if solver.solution.status == Q.QOCO_SOLVED
        # If it does converge, the point must be a genuine KKT point of the
        # problem as stated, which the oracle checks independently.
        problem = OracleProblem(
            nonconvex, [0.0, 0.0], nothing, nothing,
            sparse([1, 2, 3, 4], [1, 2, 1, 2], [-1.0, -1.0, 1.0, 1.0], 4, 2),
            [1.0, 1.0, 1.0, 1.0], 4, Int[],
        )
        check_result(problem, solver.solution; label = "nonconvex")
    end
end

# --- warm starts ------------------------------------------------------------

@testset "Warm-start coordinates transform correctly" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, _ = build(P, c, A, b, G, h, l, q; scaling_mode = :recompute)
    Q._solve!(solver)
    reference = (
        copy(solver.solution.x), copy(solver.solution.s),
        copy(solver.solution.y), copy(solver.solution.z),
    )
    # Change the data enough that the recomputed scaling is genuinely
    # different, then confirm the round trip through original units is exact
    # for all four components.
    Q.update_data!(solver; Gx = G.nzval .* 100.0)
    ws = solver.warmstart
    @test !ws.scaled
    @test isapprox(ws.x, reference[1]; rtol = 1e-12)
    @test isapprox(ws.s, reference[2]; rtol = 1e-12)
    @test isapprox(ws.y, reference[3]; rtol = 1e-12)
    @test isapprox(ws.z, reference[4]; rtol = 1e-12)

    # Converting into the new scaling and back must be the identity.
    Q._warmstart_to_scaled!(solver)
    Q.unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
    @test isapprox(solver.solution.x, reference[1]; rtol = 1e-10)
    @test isapprox(solver.solution.s, reference[2]; rtol = 1e-10)
    @test isapprox(solver.solution.y, reference[3]; rtol = 1e-10)
    @test isapprox(solver.solution.z, reference[4]; rtol = 1e-10)
end

@testset "Warm-start modes behave as documented" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    for mode in (:none, :primal, :primal_dual, :adaptive)
        solver, problem = build(P, c, A, b, G, h, l, q; warm_start_mode = mode)
        Q._solve!(solver)
        @test solver.solution.status == Q.QOCO_SOLVED
        if mode == :none
            @test !solver.warmstart.active
        else
            @test solver.warmstart.active
        end
        # A vector-only update followed by a re-solve must reach the same
        # answer whatever the reuse mode is.
        newc = c .* 1.5
        Q.update_vector_data!(solver; c = newc)
        Q._solve!(solver)
        fresh, fresh_problem = build(P, newc, A, b, G, h, l, q; warm_start_mode = mode)
        Q._solve!(fresh)
        @test solver.solution.status == Q.QOCO_SOLVED
        check_result(fresh_problem, solver.solution; label = "warm start $mode")
        @test isapprox(solver.solution.x, fresh.solution.x; atol = 1e-5)
    end
end

@testset "Warm start survives a large trust-region change" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, _ = build(P, c, A, b, G, h, l, q; warm_start_mode = :adaptive)
    Q._solve!(solver)
    # The equality forces x1 + x2 = 0.6, so any radius above 0.6/sqrt(2) keeps
    # the second-order cone feasible.
    for radius in (1e3, 0.5, 1.5)
        newh = copy(h)
        newh[3] = radius
        Q.update_vector_data!(solver; h = newh)
        Q._solve!(solver)
        fresh, problem = build(P, c, A, b, G, newh, l, q)
        Q._solve!(fresh)
        @test solver.solution.status in (Q.QOCO_SOLVED, Q.QOCO_SOLVED_INACCURATE)
        check_result(problem, solver.solution; atol = 1e-5, rtol = 1e-5, label = "radius $radius")
    end
end

@testset "Invalid manual warm starts are rejected" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, problem = build(P, c, A, b, G, h, l, q)
    @test_throws ArgumentError Q.warm_start!(solver; x = fill(NaN, 4))
    @test_throws ArgumentError Q.warm_start!(solver; x = [1.0, 2.0])
    Q.warm_start!(solver; x = zeros(4), s = zeros(5), y = zeros(1), z = zeros(5))
    Q._solve!(solver)
    @test solver.solution.status == Q.QOCO_SOLVED
    check_result(problem, solver.solution; label = "manual start")
end

# --- structure reuse --------------------------------------------------------

@testset "Fixed-pattern updates reuse the symbolic analysis" begin
    P, c, A, b, G, h, l, q = mixed_socp()
    solver, _ = build(P, c, A, b, G, h, l, q)
    Q._solve!(solver)
    factor = solver.linsys.factor
    pattern = copy(factor.L.colptr)
    for trial in 1:3
        Q.update_data!(solver; Gx = G.nzval .* (1 + 0.1 * trial), c = c .* (1 + 0.05 * trial))
        Q._solve!(solver)
        @test solver.linsys.factor === factor
        @test factor.L.colptr == pattern
    end
end
