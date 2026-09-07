# Baseline worker.
#
# This script is executed in a *separate* Julia process against a checkout of
# the audited revision, because two versions of the same package cannot be
# loaded into one session. It replays exactly the same fixtures through the
# solver API of that revision and grades the answers with exactly the same
# independent oracle, then writes the results as JSON for the comparison
# script to merge.
#
# Usage: julia --project=<env> baseline_worker.jl <fixtures.jl> <oracle.jl> <out.json>

using LinearAlgebra
using Printf
using SparseArrays
using Statistics

import JuliaQOCO

include(ARGS[1])
include(ARGS[2])
using .Fixtures
using .Oracle

const OUTPUT = ARGS[3]

quiet_settings() = JuliaQOCO.Settings{Float64}(;
    verbose = false, scaling_mode = :once, warm_start_mode = :primal_dual,
)

function grade(instance, solution)
    problem = OracleProblem(
        instance.P, instance.c, instance.A, instance.b,
        instance.G, instance.h, instance.l, instance.q,
    )
    report = oracle_report(problem, solution.x, solution.s, solution.y, solution.z)
    report.finite || return (Inf, Inf)
    residual = max(
        report.eq_residual / max(1.0, report.primal_reference),
        report.cone_residual / max(1.0, report.primal_reference),
        report.dual_residual / max(1.0, report.dual_reference),
        report.primal_cone_distance,
        report.dual_cone_distance,
    )
    return (residual, abs(report.complementarity) / report.objective_reference)
end

function replay(fixture, repetitions, tolerance)
    base = instance_at(fixture, 0)
    times = Float64[]
    passes = 0
    failures = 0
    worst_residual = 0.0
    worst_gap = 0.0
    for _ in 1:repetitions
        solver = JuliaQOCO.CoreSolver(
            base.P, base.c, base.A, base.b, base.G, base.h, base.l, base.q;
            settings = quiet_settings(),
        )
        JuliaQOCO._solve!(solver)
        for step in 1:length(fixture.sequence)
            instance = instance_at(fixture, step)
            elapsed = @timed begin
                JuliaQOCO.update_matrix_data!(
                    solver;
                    Px = instance.P.nzval, Ax = instance.A.nzval, Gx = instance.G.nzval,
                )
                JuliaQOCO.update_vector_data!(
                    solver; c = instance.c, b = instance.b, h = instance.h,
                )
                JuliaQOCO._solve!(solver)
            end
            solution = solver.solution
            residual, gap = grade(instance, solution)
            worst_residual = max(worst_residual, min(residual, 1e300))
            worst_gap = max(worst_gap, min(gap, 1e300))
            if residual <= tolerance && gap <= tolerance
                passes += 1
                push!(times, elapsed.time)
            else
                failures += 1
            end
        end
    end
    return Dict(
        "fixture" => fixture.name,
        "median_time_sec" => isempty(times) ? nothing : median(times),
        "quality_passes" => passes,
        "quality_failures" => failures,
        "worst_relative_residual" => min(worst_residual, 1e300),
        "worst_relative_gap" => min(worst_gap, 1e300),
    )
end

function main()
    fixtures = [
        build_fixture(; name = "small", horizon = 10, nx = 4, nu = 2, steps = 6),
        build_fixture(; name = "medium", horizon = 30, nx = 6, nu = 3, steps = 6),
        build_fixture(; name = "large-soc", horizon = 40, nx = 6, nu = 3, steps = 3),
    ]
    # Warm the compilation path first.
    replay(build_fixture(; name = "warmup", horizon = 3, nx = 2, nu = 1, steps = 1), 1, 1e-5)
    records = [replay(fixture, 3, 1e-5) for fixture in fixtures]

    open(OUTPUT, "w") do io
        println(io, "[")
        for (index, record) in enumerate(records)
            print(io, "  {")
            entries = [
                "\"$k\": " * (record[k] === nothing ? "null" : string(record[k]))
                for k in ("median_time_sec", "quality_passes", "quality_failures",
                          "worst_relative_residual", "worst_relative_gap")
            ]
            print(io, "\"fixture\": \"", record["fixture"], "\", ", join(entries, ", "))
            println(io, index == length(records) ? "}" : "},")
        end
        println(io, "]")
    end
    return nothing
end

main()
