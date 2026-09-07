# Deterministic replay benchmark.
#
# A fixed sequence of problem instances on a fixed sparsity pattern is replayed
# identically by every solver and configuration. Each configuration is reset to
# the same starting instance before every repetition, so no harness can evolve
# the data between samples and no two configurations ever see a different
# problem.
#
# Timings are only recorded for runs that pass an independent quality gate,
# computed from the original problem data by `test/oracle.jl`, which shares no
# code with any solver being measured. Runs that fail the gate are reported as
# quality failures instead of being silently dropped, and a configuration whose
# baseline failed the gate never produces a speedup ratio.

module Replay

using LinearAlgebra
using Printf
using SparseArrays
using Statistics

import MathOptInterface as MOI
import JuliaQOCO

include(joinpath(@__DIR__, "fixtures.jl"))
using .Fixtures
include(joinpath(@__DIR__, "adapters.jl"))
using .Adapters
include(joinpath(@__DIR__, "..", "test", "oracle.jl"))
using .Oracle

export run_replay, ReplayRecord, replay_records_to_csv, environment_summary

struct ReplayRecord
    solver::String
    workload::String
    fixture::String
    repetitions::Int
    median_time_sec::Float64
    p95_time_sec::Float64
    median_alloc_bytes::Float64
    median_iterations::Float64
    median_solver_time_sec::Float64
    quality_passes::Int
    quality_failures::Int
    worst_relative_residual::Float64
    worst_relative_gap::Float64
    note::String
end

function oracle_problem(instance::ConicInstance)
    return OracleProblem(
        instance.P, instance.c, instance.A, instance.b,
        instance.G, instance.h, instance.l, instance.q,
    )
end

# The independent gate. It never consults the solver's own residual, objective
# or status: only the returned vectors and the original data.
function quality_of(instance::ConicInstance, solution)
    problem = oracle_problem(instance)
    report = oracle_report(problem, solution.x, solution.s, solution.y, solution.z)
    report.finite || return (Inf, Inf, false)
    scale_p = max(1.0, report.primal_reference)
    scale_d = max(1.0, report.dual_reference)
    residual = max(
        report.eq_residual / scale_p,
        report.cone_residual / scale_p,
        report.dual_residual / scale_d,
        report.primal_cone_distance,
        report.dual_cone_distance,
    )
    gap = abs(report.complementarity) / report.objective_reference
    return (residual, gap, true)
end

_percentile(values, p) =
    isempty(values) ? NaN : sort(values)[clamp(ceil(Int, p * length(values)), 1, length(values))]

mutable struct Measurement
    times::Vector{Float64}
    allocations::Vector{Float64}
    iterations::Vector{Float64}
    solver_times::Vector{Float64}
    passes::Int
    failures::Int
    worst_residual::Float64
    worst_gap::Float64
end

Measurement() = Measurement(Float64[], Float64[], Float64[], Float64[], 0, 0, 0.0, 0.0)

function record!(
    measurement::Measurement,
    elapsed,
    instance::ConicInstance,
    adapter::SolverAdapter,
    tolerance::Float64,
)
    solution = solution_of(adapter)
    residual, gap, finite = quality_of(instance, solution)
    accepted = finite && residual <= tolerance && gap <= tolerance
    measurement.worst_residual = max(measurement.worst_residual, min(residual, 1e300))
    measurement.worst_gap = max(measurement.worst_gap, min(gap, 1e300))
    if accepted
        measurement.passes += 1
        push!(measurement.times, elapsed.time)
        push!(measurement.allocations, Float64(elapsed.bytes))
        stats = stats_of(adapter)
        push!(measurement.iterations, Float64(get(stats, :iterations, -1)))
        push!(measurement.solver_times, Float64(get(stats, :solve_time_sec, NaN)))
    else
        measurement.failures += 1
    end
    return accepted
end

function finish(
    measurement::Measurement,
    solver::AbstractString,
    workload::AbstractString,
    fixture::AbstractString,
    note::AbstractString = "",
)
    return ReplayRecord(
        String(solver), String(workload), String(fixture),
        measurement.passes + measurement.failures,
        isempty(measurement.times) ? NaN : median(measurement.times),
        _percentile(measurement.times, 0.95),
        isempty(measurement.allocations) ? NaN : median(measurement.allocations),
        isempty(measurement.iterations) ? NaN : median(measurement.iterations),
        isempty(measurement.solver_times) ? NaN : median(measurement.solver_times),
        measurement.passes,
        measurement.failures,
        measurement.worst_residual,
        measurement.worst_gap,
        String(note),
    )
end

# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------

function workload_fresh(adapter_factory, fixture::Fixture, repetitions::Int, tolerance::Float64)
    measurement = Measurement()
    base = instance_at(fixture, 0)
    for _ in 1:repetitions
        adapter = adapter_factory()
        elapsed = @timed begin
            setup!(adapter, base)
            solve!(adapter)
        end
        record!(measurement, elapsed, base, adapter, tolerance)
    end
    return measurement
end

function workload_unchanged(adapter_factory, fixture::Fixture, repetitions::Int, tolerance::Float64)
    measurement = Measurement()
    base = instance_at(fixture, 0)
    adapter = adapter_factory()
    setup!(adapter, base)
    solve!(adapter)
    for _ in 1:repetitions
        elapsed = @timed solve!(adapter)
        record!(measurement, elapsed, base, adapter, tolerance)
    end
    return measurement
end

# Replay the whole precomputed sequence, resetting to the same starting point
# before each repetition.
function workload_sequence(
    adapter_factory,
    fixture::Fixture,
    repetitions::Int,
    tolerance::Float64;
    matrices::Bool,
    objective::Bool,
    vectors::Bool,
)
    measurement = Measurement()
    base = instance_at(fixture, 0)
    for _ in 1:repetitions
        adapter = adapter_factory()
        setup!(adapter, base)
        solve!(adapter)
        for step in 1:length(fixture.sequence)
            # A vector-only or objective-only update leaves the other blocks at
            # their starting values, so the problem actually solved is a
            # mixture. The mixture is what the adapter is given and what the
            # oracle checks, which keeps a solver that rebuilds from scratch on
            # exactly the same problem as one that updates in place.
            effective = _effective_instance(
                base, instance_at(fixture, step); matrices, objective, vectors,
            )
            elapsed = @timed begin
                update!(adapter, effective; matrices, objective, vectors)
                solve!(adapter)
            end
            record!(measurement, elapsed, effective, adapter, tolerance)
        end
    end
    return measurement
end

function _effective_instance(
    base::ConicInstance,
    instance::ConicInstance;
    matrices::Bool,
    objective::Bool,
    vectors::Bool,
)
    matrices && objective && vectors && return instance
    return ConicInstance{Float64}(
        matrices && objective ? instance.P : base.P,
        objective ? instance.c : base.c,
        matrices ? instance.A : base.A,
        vectors ? instance.b : base.b,
        matrices ? instance.G : base.G,
        vectors ? instance.h : base.h,
        base.l,
        base.q,
    )
end

# A local change: only a fraction of the matrix entries move. The changed
# fraction is recorded so the number can be interpreted.
function workload_sparse_subset(
    adapter_factory,
    fixture::Fixture,
    repetitions::Int,
    tolerance::Float64,
    fraction::Float64,
)
    measurement = Measurement()
    base = instance_at(fixture, 0)
    stride = max(1, round(Int, inv(fraction)))
    for _ in 1:repetitions
        adapter = adapter_factory()
        setup!(adapter, base)
        solve!(adapter)
        for step in 1:length(fixture.sequence)
            target = instance_at(fixture, step)
            values = copy(base.A.nzval)
            @inbounds for k in 1:stride:length(values)
                values[k] = target.A.nzval[k]
            end
            mixed = ConicInstance{Float64}(
                base.P, base.c,
                SparseMatrixCSC(size(base.A)..., copy(base.A.colptr), copy(base.A.rowval), values),
                base.b, base.G, base.h, base.l, base.q,
            )
            elapsed = @timed begin
                update!(adapter, mixed; matrices = true, objective = false, vectors = false)
                solve!(adapter)
            end
            record!(measurement, elapsed, mixed, adapter, tolerance)
        end
    end
    return measurement
end

# A coefficient inside the reserved pattern is driven to zero and back. This is
# a correctness workload as much as a timing one: it must not rebuild.
function workload_zero_crossing(
    adapter_factory,
    fixture::Fixture,
    repetitions::Int,
    tolerance::Float64,
)
    measurement = Measurement()
    base = instance_at(fixture, 0)
    # Pick an off-diagonal dynamics coefficient.
    position = findfirst(k -> base.A.nzval[k] != 0 && abs(base.A.nzval[k]) != 1, 1:nnz(base.A))
    position === nothing && (position = 1)
    for _ in 1:repetitions
        adapter = adapter_factory()
        setup!(adapter, base)
        solve!(adapter)
        for value in (0.0, base.A.nzval[position], 0.0, base.A.nzval[position])
            values = copy(base.A.nzval)
            values[position] = value
            mixed = ConicInstance{Float64}(
                base.P, base.c,
                SparseMatrixCSC(size(base.A)..., copy(base.A.colptr), copy(base.A.rowval), values),
                base.b, base.G, base.h, base.l, base.q,
            )
            elapsed = @timed begin
                update!(adapter, mixed; matrices = true, objective = false, vectors = false)
                solve!(adapter)
            end
            record!(measurement, elapsed, mixed, adapter, tolerance)
        end
    end
    return measurement
end

# A step change far larger than a normal sequential-convex-programming update:
# the trust radius collapses and the penalty weights jump.
function workload_large_change(
    adapter_factory,
    fixture::Fixture,
    repetitions::Int,
    tolerance::Float64,
)
    measurement = Measurement()
    base = instance_at(fixture, 0)
    # A hundredfold penalty swing and a more than fourfold trust-radius swing.
    # The radius stays above the value the unconstrained optimum needs, so this
    # measures robustness to a large step rather than the response to an
    # infeasible subproblem: below that value the trust region simply cannot be
    # satisfied, and every solver correctly refuses it.
    base_radius = _trust_radius(base)
    shrunk = ConicInstance{Float64}(
        base.P .* 100.0, base.c .* 100.0, base.A, base.b, base.G,
        _with_trust_radius(base, 0.9 * base_radius), base.l, base.q,
    )
    grown = ConicInstance{Float64}(
        base.P .* 0.01, base.c .* 0.01, base.A, base.b, base.G,
        _with_trust_radius(base, 4.0 * base_radius), base.l, base.q,
    )
    for _ in 1:repetitions
        adapter = adapter_factory()
        setup!(adapter, base)
        solve!(adapter)
        for instance in (shrunk, grown, base, shrunk)
            elapsed = @timed begin
                update!(adapter, instance; matrices = true, objective = true, vectors = true)
                solve!(adapter)
            end
            record!(measurement, elapsed, instance, adapter, tolerance)
        end
    end
    return measurement
end

# The trust region is the last cone block, and its head row carries the radius.
_trust_head(instance::ConicInstance) = length(instance.h) - last(instance.q) + 1
_trust_radius(instance::ConicInstance) = instance.h[_trust_head(instance)]

function _with_trust_radius(instance::ConicInstance, radius::Float64)
    h = copy(instance.h)
    h[_trust_head(instance)] = radius
    return h
end

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

function environment_summary()
    return Dict(
        "julia_version" => string(VERSION),
        "cpu" => Sys.cpu_info()[1].model,
        "cpu_threads" => Sys.CPU_THREADS,
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "os" => string(Sys.KERNEL, " ", Sys.MACHINE),
        "juliaqoco_version" => string(pkgversion(JuliaQOCO)),
        "moi_version" => string(pkgversion(MOI)),
    )
end

function solver_configurations(; include_references::Bool = true)
    configurations = Pair{String,Any}[]
    push!(configurations, "juliaqoco-native-reuse" =>
        () -> NativeAdapter("juliaqoco-native-reuse"; scaling_mode = :once, warm_start_mode = :primal_dual))
    push!(configurations, "juliaqoco-native-adaptive" =>
        () -> NativeAdapter("juliaqoco-native-adaptive"; scaling_mode = :once, warm_start_mode = :adaptive))
    push!(configurations, "juliaqoco-native-cold" =>
        () -> NativeAdapter("juliaqoco-native-cold"; scaling_mode = :once, warm_start_mode = :none))
    push!(configurations, "juliaqoco-native-rescaled" =>
        () -> NativeAdapter("juliaqoco-native-rescaled"; scaling_mode = :recompute, warm_start_mode = :primal_dual))
    push!(configurations, "juliaqoco-moi" =>
        () -> MoiAdapter("juliaqoco-moi", () -> JuliaQOCO.Optimizer(; verbose = false)))
    push!(configurations, "juliaqoco-moi-rebuild" =>
        () -> MoiAdapter(
            "juliaqoco-moi-rebuild",
            () -> JuliaQOCO.Optimizer(; verbose = false);
            rebuild_each_solve = true,
        ))
    include_references || return configurations
    if clarabel_available()
        push!(configurations, "clarabel-update" => () -> ClarabelAdapter("clarabel-update"; reuse = true))
        push!(configurations, "clarabel-fresh" => () -> ClarabelAdapter("clarabel-fresh"; reuse = false))
    end
    if Adapters.c_qoco_available()
        push!(configurations, "c-qoco-moi-fresh" =>
            () -> MoiAdapter(
                "c-qoco-moi-fresh", Adapters.c_qoco_optimizer_factory();
                rebuild_each_solve = true,
            ))
    end
    return configurations
end

"""
    run_replay(; fixtures, repetitions, tolerance, include_references)

Run every workload for every solver configuration and return the records.
"""
function run_replay(;
    fixtures::Vector{<:Fixture} = [
        build_fixture(; name = "small", horizon = 10, nx = 4, nu = 2, steps = 6),
        build_fixture(; name = "medium", horizon = 30, nx = 6, nu = 3, steps = 6),
        build_fixture(; name = "large-soc", horizon = 40, nx = 6, nu = 3, steps = 3),
    ],
    repetitions::Int = 3,
    tolerance::Float64 = 1e-5,
    include_references::Bool = true,
    verbose::Bool = true,
    solvers::Union{Nothing,Vector{String}} = nothing,
    workloads::Union{Nothing,Vector{String}} = nothing,
)
    records = ReplayRecord[]
    configurations = solver_configurations(; include_references)
    solvers === nothing || filter!(pair -> first(pair) in solvers, configurations)
    # First-use latency is a real cost, but it is a different cost from steady
    # state and mixing the two into one median hides both. Every configuration
    # is compiled once on a tiny instance first, and the latency is reported
    # separately.
    warmup = build_fixture(; name = "warmup", horizon = 3, nx = 2, nu = 1, steps = 1)
    for (name, factory) in configurations
        latency = @elapsed begin
            adapter = factory()
            setup!(adapter, instance_at(warmup, 0))
            solve!(adapter)
            supports_fixed_pattern_update(adapter) &&
                update!(adapter, instance_at(warmup, 1))
            solve!(adapter)
            solution_of(adapter)
            stats_of(adapter)
        end
        push!(records, ReplayRecord(
            name, "first_use_latency", warmup.name, 1, latency, latency,
            NaN, NaN, NaN, 1, 0, NaN, NaN,
            "compilation and first-use cost on a tiny instance, excluded from every other row",
        ))
        verbose && @printf("  %-26s first-use latency %8.3f s\n", name, latency)
    end
    for fixture in fixtures
        verbose && println("fixture ", fixture.name, ": ", pattern_summary(fixture))
        # A large global trust-region cone costs hundreds of milliseconds per
        # solve, so repeating it as often as a small one would dominate the
        # whole run for no extra information.
        fixture_repetitions =
            length(fixture.base.c) > 400 ? 1 : repetitions
        for (name, factory) in configurations
            fixed_pattern = supports_fixed_pattern_update(factory())
            workload_list = Any[
                ("fresh_setup_and_solve", () -> workload_fresh(factory, fixture, fixture_repetitions, tolerance), ""),
                ("unchanged_resolve", () -> workload_unchanged(factory, fixture, fixture_repetitions, tolerance),
                 "reuse overhead only; no linearization changed"),
                ("vector_updates", () -> workload_sequence(
                    factory, fixture, fixture_repetitions, tolerance;
                    matrices = false, objective = false, vectors = true), ""),
                ("objective_and_vector_updates", () -> workload_sequence(
                    factory, fixture, fixture_repetitions, tolerance;
                    matrices = false, objective = true, vectors = true), ""),
                ("full_matrix_updates", () -> workload_sequence(
                    factory, fixture, fixture_repetitions, tolerance;
                    matrices = true, objective = false, vectors = true), ""),
                ("full_update", () -> workload_sequence(
                    factory, fixture, fixture_repetitions, tolerance;
                    matrices = true, objective = true, vectors = true), ""),
                ("sparse_subset_10pct", () -> workload_sparse_subset(
                    factory, fixture, fixture_repetitions, tolerance, 0.1),
                 "about 10 percent of nnz(A) changed"),
                ("reserved_zero_crossing", () -> workload_zero_crossing(
                    factory, fixture, fixture_repetitions, tolerance), ""),
                ("large_penalty_and_radius_change", () -> workload_large_change(
                    factory, fixture, fixture_repetitions, tolerance), ""),
            ]
            for (workload, run, note) in workload_list
                workloads === nothing || workload in workloads || continue
                if !fixed_pattern && workload != "fresh_setup_and_solve" &&
                   workload != "unchanged_resolve"
                    note = isempty(note) ? "rebuilt from scratch each solve" :
                           string(note, "; rebuilt from scratch each solve")
                end
                measurement = try
                    run()
                catch error
                    verbose && @warn "workload failed" solver = name workload = workload error = error
                    Measurement()
                end
                record = finish(measurement, name, workload, fixture.name, note)
                push!(records, record)
                verbose && @printf(
                    "  %-26s %-32s median=%8.3f ms  q_pass=%d q_fail=%d\n",
                    name, workload, 1e3 * record.median_time_sec,
                    record.quality_passes, record.quality_failures,
                )
            end
        end
    end
    return records
end

function replay_records_to_csv(records::Vector{ReplayRecord})
    io = IOBuffer()
    println(io, join(string.(fieldnames(ReplayRecord)), ","))
    for record in records
        values = map(fieldnames(ReplayRecord)) do field
            value = getfield(record, field)
            value isa AbstractString ? string('"', value, '"') : string(value)
        end
        println(io, join(values, ","))
    end
    return String(take!(io))
end

end # module Replay
