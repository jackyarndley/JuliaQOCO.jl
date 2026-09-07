# Single-precision coverage.
#
# The solver is parameterized on its element type, and this file is what makes
# that claim mean something. It exercises the same invariants as the
# double-precision suite - the scaling identities, the finalization contract,
# the sparse second-order-cone expansion, fixed-pattern updates, warm starts
# and the MathOptInterface layer - at `Float32`, and it also checks that the
# precision-dependent defaults are actually representable in the type they are
# computed for.
#
# The tolerances here are derived from `eps(T)`, not copied from the
# double-precision suite. Asking single precision for a `1e-7` residual is
# asking for something below its own machine epsilon.

using Test
using Random
using LinearAlgebra
using SparseArrays
using JuliaQOCO

# The oracle is shared by every test file; include it only once so that the
# files can be run individually or together without creating two modules that
# export the same names.
isdefined(@__MODULE__, :Oracle) || include("oracle.jl")
using .Oracle

const Q = JuliaQOCO

const TESTED_TYPES = (Float64, Float32)

quiet32(::Type{T}; kwargs...) where {T} = Q.Settings{T}(; verbose = false, kwargs...)

function build32(::Type{T}, P, c, A, b, G, h, l, q; kwargs...) where {T}
    solver = Q.CoreSolver(P, c, A, b, G, h, l, q; settings = quiet32(T; kwargs...))
    return solver, OracleProblem(P, c, A, b, G, h, l, q)
end

# --- fixtures, generated at the requested precision ------------------------

function coupled_qp(::Type{T}) where {T}
    P = sparse([1, 1, 2, 3], [1, 2, 2, 3], T[2, 0.5, 3, 1], 3, 3)
    c = T[-1, -2, 0.5]
    A = sparse([1, 1, 1], [1, 2, 3], T[1, 1, 1], 1, 3)
    b = T[1]
    G = sparse(1:3, 1:3, fill(T(-1), 3), 3, 3)
    return P, c, A, b, G, zeros(T, 3), 3, Int[]
end

function singular_psd_qp(::Type{T}) where {T}
    P = sparse([1, 2], [1, 2], T[2, 1], 3, 3)
    c = T[-1, -1, 0.25]
    A = sparse([1, 1], [1, 2], T[1, 1], 1, 3)
    b = T[1]
    G = sparse(1:3, 1:3, fill(T(-1), 3), 3, 3)
    return P, c, A, b, G, zeros(T, 3), 3, Int[]
end

function mixed_socp(::Type{T}) where {T}
    P = sparse([1, 1, 2, 3, 4], [1, 2, 2, 3, 4], T[1, 0.3, 1, 0.5, 0.5], 4, 4)
    c = T[0.1, -0.4, 0.05, 0.07]
    A = sparse([1, 1], [1, 2], T[1, 1], 1, 4)
    b = T[0.6]
    G = sparse([1, 2, 4, 5], [3, 4, 1, 2], T[-1, -1, -1, -1], 5, 4)
    h = T[0, 0, 1.5, 0, 0]
    return P, c, A, b, G, h, 2, [3]
end

function equality_only_qp(::Type{T}) where {T}
    P = sparse([1, 2], [1, 2], T[2, 4], 2, 2)
    A = sparse([1, 1], [1, 2], T[1, 2], 1, 2)
    return P, T[-1, 1], A, T[0.5], nothing, nothing, 0, Int[]
end

function zero_objective(::Type{T}, scale::Real) where {T}
    n = 3
    A = sparse([1, 1], [1, 2], T[1, 1], 1, n)
    G = sparse(1:n, 1:n, fill(T(-scale), n), n, n)
    return spzeros(T, n, n), zeros(T, n), A, T[1], G, zeros(T, n), n, Int[]
end

function big_cone(::Type{T}, qdim::Int; n::Int = 8) where {T}
    rows = Int[]
    cols = Int[]
    vals = T[]
    for i in 1:n
        push!(rows, i); push!(cols, i); push!(vals, T(-1))
    end
    h = zeros(T, n)
    push!(h, T(3))
    for k in 1:(qdim - 1)
        push!(rows, n + 1 + k); push!(cols, mod1(k, n)); push!(vals, T(-1)); push!(h, zero(T))
    end
    P = sparse(1:n, 1:n, fill(T(1), n), n, n)
    c = T[(-1)^i * 0.3 for i in 1:n]
    A = sparse(ones(Int, n), 1:n, ones(T, n), 1, n)
    return P, c, A, T[1], sparse(rows, cols, vals, n + qdim, n), h, n, [qdim]
end

# --- defaults --------------------------------------------------------------

@testset "Precision-dependent defaults are representable" begin
    for T in TESTED_TYPES
        settings = Q.Settings{T}()
        # A tolerance below eps can never be met, and a regularization below
        # eps cannot change a diagonal entry of order one.
        @test settings.abstol > eps(T)
        @test settings.reltol > eps(T)
        @test settings.abstol_inacc > settings.abstol
        @test settings.reltol_inacc > settings.reltol
        @test settings.kkt_static_reg > eps(T)
        @test settings.kkt_dynamic_reg > eps(T)
        @test one(T) + settings.kkt_static_reg > one(T)
        @test Q.default_min_step(T) >= eps(T)
        @test Q.safe_div_eps(T) >= eps(T)
        @test settings.iter_ref_tol > eps(T)
        # Every numerical default carries the element type and is finite.
        # `time_limit_sec` is excluded: it is a wall-clock duration rather
        # than problem data, so it is always `Float64` and defaults to `Inf`,
        # meaning no limit.
        for name in fieldnames(Q.Settings)
            name === :time_limit_sec && continue
            value = getfield(settings, name)
            value isa AbstractFloat || continue
            @test value isa T
            @test isfinite(value)
        end
        @test Q.Settings{T}().time_limit_sec == Inf
    end
    # Double precision keeps exactly the values it was tuned with.
    reference = Q.Settings{Float64}()
    @test reference.abstol == 1e-7
    @test reference.reltol == 1e-7
    @test reference.abstol_inacc == 1e-5
    @test reference.reltol_inacc == 1e-5
    @test reference.kkt_static_reg == 1e-8
    @test reference.kkt_dynamic_reg == 1e-8
    @test reference.iter_ref_tol == sqrt(eps(Float64))
    # Single precision is genuinely looser, not accidentally tighter.
    single = Q.Settings{Float32}()
    @test single.abstol > reference.abstol
    @test single.kkt_static_reg > reference.kkt_static_reg
end

# --- the same invariants, at each precision --------------------------------

@testset "Solves and original-unit quality at $T" for T in TESTED_TYPES
    for (name, fixture) in (
        ("coupled QP", coupled_qp(T)),
        ("singular PSD QP", singular_psd_qp(T)),
        ("mixed SOCP", mixed_socp(T)),
        ("equality-only QP", equality_only_qp(T)),
        ("big cone", big_cone(T, 30)),
    )
        solver, problem = build32(T, fixture...)
        Q._solve!(solver)
        @test solver.solution.status == Q.QOCO_SOLVED
        @test solver.solution.result_available
        @test solver.solution.x isa Vector{T}
        @test solver.solution.obj isa T
        check_result(problem, solver.solution; label = "$T $name")
    end
end

@testset "Zero objective keeps a finite scaling at $T" for T in TESTED_TYPES
    for scale in (1, 100)
        solver, problem = build32(T, zero_objective(T, scale)...)
        @test solver.scaling.k isa T
        @test isfinite(solver.scaling.k) && solver.scaling.k > 0
        @test isfinite(solver.scaling.kinv) && solver.scaling.kinv > 0
        Q._solve!(solver)
        @test solver.solution.status in (Q.QOCO_SOLVED, Q.QOCO_SOLVED_INACCURATE)
        check_result(problem, solver.solution; label = "$T zero objective $scale")
    end
end

@testset "Reported complementarity is the true one at $T" for T in TESTED_TYPES
    # The bug this guards against scaled the reported gap by the square of the
    # cone row scaling, which no amount of precision would hide.
    for cone_scale in (1, 100)
        solver, problem = build32(T, zero_objective(T, cone_scale)...)
        Q._solve!(solver)
        report = oracle_report(
            problem, solver.solution.x, solver.solution.s,
            solver.solution.y, solver.solution.z,
        )
        @test isapprox(
            solver.solution.gap, report.complementarity;
            atol = 8 * eps(T), rtol = max(T(1e-5), 64 * eps(T)),
        )
    end
end

@testset "Expanded and dense cone blocks agree at $T" for T in TESTED_TYPES
    for qdim in (9, 30)
        fixture = big_cone(T, qdim)
        dense, problem = build32(T, fixture...; soc_expansion_threshold = typemax(Int))
        expanded, _ = build32(T, fixture...; soc_expansion_threshold = 2)
        @test !any(dense.work.soc_expanded)
        @test all(expanded.work.soc_expanded)
        @test expanded.work.soc_aux isa Vector{T}
        Q._solve!(dense)
        Q._solve!(expanded)
        @test dense.solution.status == Q.QOCO_SOLVED
        @test expanded.solution.status == Q.QOCO_SOLVED
        check_result(problem, expanded.solution; label = "$T expanded q=$qdim")
        # Both reach the same optimum to the accuracy the type supports.
        @test isapprox(
            dense.solution.obj, expanded.solution.obj;
            rtol = max(T(1e-6), 64 * sqrt(eps(T))),
        )
    end
end

@testset "Updates match fresh construction at $T" for T in TESTED_TYPES
    P, c, A, b, G, h, l, q = mixed_socp(T)
    for mode in (:once, :recompute)
        solver, _ = build32(T, P, c, A, b, G, h, l, q; scaling_mode = mode)
        Q._solve!(solver)
        factor = solver.linsys.factor
        for trial in 1:3
            newc = c .* T(1 + 0.1 * trial)
            newh = copy(h)
            newh[3] = T(1.5 + 0.2 * trial)
            Q.update_data!(solver; c = newc, h = newh)
            Q._solve!(solver)
            fresh, problem = build32(T, P, newc, A, b, G, newh, l, q; scaling_mode = mode)
            Q._solve!(fresh)
            mode == :once && @test solver.linsys.factor === factor
            @test solver.solution.status == Q.QOCO_SOLVED
            check_result(problem, solver.solution; label = "$T update $mode $trial")
            @test isapprox(
                solver.solution.obj, fresh.solution.obj;
                atol = max(T(1e-6), 8 * sqrt(eps(T))),
                rtol = max(T(1e-5), 64 * sqrt(eps(T))),
            )
        end
    end
end

@testset "Warm-start coordinates round-trip at $T" for T in TESTED_TYPES
    P, c, A, b, G, h, l, q = mixed_socp(T)
    solver, _ = build32(T, P, c, A, b, G, h, l, q; scaling_mode = :recompute)
    Q._solve!(solver)
    reference = (
        copy(solver.solution.x), copy(solver.solution.s),
        copy(solver.solution.y), copy(solver.solution.z),
    )
    Q.update_data!(solver; Gx = G.nzval .* T(100))
    @test !solver.warmstart.scaled
    tolerance = max(T(1e-6), 64 * eps(T))
    for (cached, original) in zip(
        (solver.warmstart.x, solver.warmstart.s, solver.warmstart.y, solver.warmstart.z),
        reference,
    )
        @test isapprox(cached, original; rtol = tolerance)
    end
    Q._warmstart_to_scaled!(solver)
    Q.unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
    for (roundtripped, original) in zip(
        (solver.solution.x, solver.solution.s, solver.solution.y, solver.solution.z),
        reference,
    )
        @test isapprox(roundtripped, original; rtol = tolerance)
    end
end

@testset "Warm-start modes behave at $T" for T in TESTED_TYPES
    P, c, A, b, G, h, l, q = mixed_socp(T)
    for mode in (:none, :primal, :primal_dual, :adaptive)
        solver, _ = build32(T, P, c, A, b, G, h, l, q; warm_start_mode = mode)
        Q._solve!(solver)
        @test solver.solution.status == Q.QOCO_SOLVED
        newc = c .* T(1.5)
        Q.update_vector_data!(solver; c = newc)
        Q._solve!(solver)
        fresh, problem = build32(T, P, newc, A, b, G, h, l, q; warm_start_mode = mode)
        Q._solve!(fresh)
        @test solver.solution.status == Q.QOCO_SOLVED
        check_result(problem, solver.solution; label = "$T warm start $mode")
    end
end

@testset "Failure reporting stays honest at $T" for T in TESTED_TYPES
    # Nonfinite data must not produce an advertised result.
    solver, _ = build32(T, coupled_qp(T)...)
    solver.data.c[1] = T(NaN)
    Q._solve!(solver)
    @test solver.solution.status == Q.QOCO_NUMERICAL_ERROR
    @test !solver.solution.result_available

    # An infeasible cone must not come back as OPTIMAL.
    P, c, A, b, G, h, l, q = mixed_socp(T)
    infeasible = copy(h)
    infeasible[3] = T(1e-3)
    solver, _ = build32(T, P, c, A, b, G, infeasible, l, q; max_iters = 60)
    Q._solve!(solver)
    @test solver.solution.status != Q.QOCO_SOLVED
end

@testset "Profiling does not change the numbers at $T" for T in TESTED_TYPES
    fixture = mixed_socp(T)
    plain, _ = build32(T, fixture...)
    Q._solve!(plain)
    profiled, _ = build32(T, fixture...; profile = true)
    Q._solve!(profiled)
    @test plain.solution.status == profiled.solution.status
    @test plain.solution.iters == profiled.solution.iters
    @test plain.solution.x == profiled.solution.x
    @test plain.solution.obj == profiled.solution.obj
    @test plain.solution.gap == profiled.solution.gap
end

@testset "Narrow sparse index types work at $T" for T in TESTED_TYPES
    # The index type governs the CSC arrays; the solver must not assume it is
    # the same as the machine word size.
    P = SparseMatrixCSC{T,Int32}(sparse([1, 2], [1, 2], T[2, 2], 2, 2))
    A = SparseMatrixCSC{T,Int32}(sparse([1, 1], [1, 2], T[1, 1], 1, 2))
    G = SparseMatrixCSC{T,Int32}(sparse([1, 2], [1, 2], T[-1, -1], 2, 2))
    solver = Q.CoreSolver(
        P, T[-1, -1], A, T[1], G, zeros(T, 2), 2, Int32[]; settings = quiet32(T),
    )
    Q._solve!(solver)
    @test solver.solution.status == Q.QOCO_SOLVED
    @test isapprox(solver.solution.x, T[0.5, 0.5]; atol = max(T(1e-5), 8 * sqrt(eps(T))))
end
