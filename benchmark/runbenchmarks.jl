# Benchmark driver.
#
#   julia --project=benchmark benchmark/runbenchmarks.jl
#   julia --project=benchmark benchmark/runbenchmarks.jl --quick
#   julia --project=benchmark benchmark/runbenchmarks.jl --no-references
#
# Writes machine-readable results and a short Markdown report under
# `benchmark/results/`, which is output and not part of the package.

using Dates
using JSON
using LinearAlgebra
using Printf
using Statistics

import MathOptInterface as MOI
import JuliaQOCO

include(joinpath(@__DIR__, "replay.jl"))
include(joinpath(@__DIR__, "scp_nonlinear.jl"))

using .Replay
using .Replay.Fixtures
using .NonlinearSCP

const RESULTS_DIRECTORY = joinpath(@__DIR__, "results")

function scp_solver_configurations(include_references::Bool)
    configurations = Any[
        ("juliaqoco-direct", () -> JuliaQOCO.Optimizer(; verbose = false), true),
        ("juliaqoco-cached-model", () -> JuliaQOCO.Optimizer(; verbose = false), false),
    ]
    include_references || return configurations
    if Replay.Adapters.clarabel_available()
        push!(configurations, ("clarabel", Replay.Adapters.clarabel_optimizer_factory(), false))
    end
    if Replay.Adapters.c_qoco_available()
        push!(configurations, ("c-qoco", Replay.Adapters.c_qoco_optimizer_factory(), false))
    end
    return configurations
end

function run_scp_comparison(; include_references::Bool = true, verbose::Bool = true)
    results = SCPResult[]
    for (name, factory, direct) in scp_solver_configurations(include_references)
        # Warm every compilation path on a tiny instance first, so the reported
        # numbers are steady state rather than first-use latency.
        latency = @elapsed run_scp(
            factory; solver_name = name, horizon = 6, max_outer = 2, direct = direct,
        )
        result = run_scp(factory; solver_name = name, direct = direct)
        verbose && @printf(
            "  %-22s converged=%-5s accepted=%2d rejected=%2d defect=%.3e endpoint=%.3e effort=%.6f subproblem=%.4f s first_use=%.2f s\n",
            name, result.converged, result.accepted_steps, result.rejected_steps,
            result.max_dynamics_defect, result.endpoint_error, result.control_effort,
            result.subproblem_time_sec, latency,
        )
        push!(results, result)
    end
    return results
end

function scp_records(results::Vector{SCPResult})
    return [
        Dict(
            "solver" => result.solver,
            "converged" => result.converged,
            "outer_iterations" => result.outer_iterations,
            "accepted_steps" => result.accepted_steps,
            "rejected_steps" => result.rejected_steps,
            "subproblem_solves" => result.subproblem_solves,
            "total_time_sec" => result.total_time_sec,
            "subproblem_time_sec" => result.subproblem_time_sec,
            "model_update_time_sec" => result.model_update_time_sec,
            "median_subproblem_time_sec" => isempty(result.subproblem_times) ? NaN :
                                            median(result.subproblem_times),
            "control_effort" => result.control_effort,
            "max_dynamics_defect" => result.max_dynamics_defect,
            "endpoint_error" => result.endpoint_error,
            "max_thrust_violation" => result.max_thrust_violation,
            "statuses" => result.subproblem_statuses,
        )
        for result in results
    ]
end

# JSON has no representation for NaN or an infinity. A missing measurement is
# written as null rather than being quietly turned into a number.
_json_safe(value::AbstractFloat) = isfinite(value) ? value : nothing
_json_safe(value::AbstractDict) = Dict(string(k) => _json_safe(v) for (k, v) in value)
_json_safe(value::AbstractVector) = [_json_safe(v) for v in value]
_json_safe(value) = value

function write_report(
    directory::AbstractString,
    environment::Dict,
    fixtures,
    replay_records::Vector{ReplayRecord},
    scp_results::Vector{SCPResult},
)
    mkpath(directory)
    write(joinpath(directory, "replay.csv"), replay_records_to_csv(replay_records))
    payload = Dict(
        "generated" => string(now()),
        "environment" => environment,
        "fixtures" => [Dict(pairs(pattern_summary(f))) for f in fixtures],
        "replay" => [Dict(string(k) => getfield(r, k) for k in fieldnames(ReplayRecord))
                     for r in replay_records],
        "scp" => scp_records(scp_results),
    )
    open(joinpath(directory, "results.json"), "w") do io
        JSON.print(io, _json_safe(payload), 2)
    end

    io = IOBuffer()
    println(io, "# JuliaQOCO benchmark results\n")
    println(io, "Generated ", now(), ".\n")
    println(io, "## Environment\n")
    for key in sort(collect(keys(environment)))
        println(io, "- `", key, "`: ", environment[key])
    end
    println(io, "\n## Fixtures\n")
    println(io, "| fixture | vars | eq rows | cone rows | orthant | SOCs | largest SOC | nnz(P) | nnz(A) | nnz(G) | steps |")
    println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for fixture in fixtures
        s = pattern_summary(fixture)
        @printf(io, "| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d |\n",
            s.name, s.variables, s.equality_rows, s.cone_rows, s.orthant_rows,
            s.soc_count, s.soc_max, s.nnz_P, s.nnz_A, s.nnz_G, s.steps)
    end

    println(io, "\n## Deterministic replay\n")
    println(io, "Timings are reported only for runs that passed an independent quality gate ")
    println(io, "computed from the original problem data. A row with `quality_failures > 0` ")
    println(io, "did not meet the gate on every replayed step, and a row whose median is ")
    println(io, "`NaN` passed the gate on none of them.\n")
    for fixture in fixtures
        rows = filter(r -> r.fixture == fixture.name, replay_records)
        isempty(rows) && continue
        println(io, "### Fixture `", fixture.name, "`\n")
        println(io, "| workload | solver | median (ms) | p95 (ms) | alloc (KiB) | IPM iters | quality pass/fail | worst rel. residual | note |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | :---: | ---: | --- |")
        for workload in unique(r.workload for r in rows)
            for record in filter(r -> r.workload == workload, rows)
                @printf(io, "| %s | %s | %s | %s | %s | %s | %d/%d | %s | %s |\n",
                    record.workload, record.solver,
                    _format(1e3 * record.median_time_sec),
                    _format(1e3 * record.p95_time_sec),
                    _format(record.median_alloc_bytes / 1024),
                    _format(record.median_iterations),
                    record.quality_passes, record.quality_failures,
                    _format(record.worst_relative_residual),
                    record.note)
            end
        end
        println(io)
    end

    println(io, "## End-to-end nonlinear sequential convex programming\n")
    println(io, "The outer loop is identical for every solver: same dynamics, same ")
    println(io, "discretization, same trust-region and penalty policy. The final answer is ")
    println(io, "checked by re-propagating the returned controls through the nonlinear ")
    println(io, "dynamics with a finer integrator.\n")
    println(io, "| solver | converged | accepted | rejected | solves | subproblem total (s) | median solve (ms) | model update (s) | control effort | dynamics defect | endpoint error | thrust violation |")
    println(io, "| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in scp_results
        @printf(io, "| %s | %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s |\n",
            result.solver, result.converged ? "yes" : "no",
            result.accepted_steps, result.rejected_steps, result.subproblem_solves,
            _format(result.subproblem_time_sec),
            _format(1e3 * (isempty(result.subproblem_times) ? NaN : median(result.subproblem_times))),
            _format(result.model_update_time_sec),
            _format(result.control_effort),
            _format(result.max_dynamics_defect),
            _format(result.endpoint_error),
            _format(result.max_thrust_violation))
    end

    println(io, "\n## Limitations\n")
    println(io, "- Every number here comes from one machine and one Julia version; see the")
    println(io, "  environment table above.")
    println(io, "- The replay workloads isolate solver cost on identical problems. The")
    println(io, "  nonlinear loop measures the outcome that actually matters, but small")
    println(io, "  numerical differences can send two solvers down different outer")
    println(io, "  trajectories, so its per-solve times are not a controlled comparison.")
    println(io, "- The C QOCO comparison goes through its MathOptInterface wrapper and")
    println(io, "  rebuilds each solve; it is a wrapper-level number, not a measurement of")
    println(io, "  the C core in isolation.")
    println(io, "- No solver here is asked to certify infeasibility, and JuliaQOCO does not")
    println(io, "  implement infeasibility certificates at all.")

    write(joinpath(directory, "report.md"), String(take!(io)))
    return nothing
end

_format(value) =
    value isa Integer ? string(value) :
    (isnan(value) ? "n/a" : (abs(value) >= 1e-3 && abs(value) < 1e5 ?
                             @sprintf("%.4g", value) : @sprintf("%.3e", value)))

function main(arguments::Vector{String} = ARGS)
    quick = "--quick" in arguments
    include_references = !("--no-references" in arguments)
    fixtures = quick ?
        [build_fixture(; name = "small", horizon = 8, nx = 4, nu = 2, steps = 3)] :
        [
            build_fixture(; name = "small", horizon = 10, nx = 4, nu = 2, steps = 6),
            build_fixture(; name = "medium", horizon = 30, nx = 6, nu = 3, steps = 6),
            build_fixture(; name = "large-soc", horizon = 40, nx = 6, nu = 3, steps = 3),
        ]
    repetitions = quick ? 1 : 3

    println("Deterministic replay benchmark")
    replay_records = run_replay(;
        fixtures = fixtures,
        repetitions = repetitions,
        include_references = include_references,
    )

    println("\nEnd-to-end nonlinear sequential convex programming benchmark")
    scp_results = run_scp_comparison(; include_references = include_references)

    environment = merge(environment_summary(), scp_environment())
    write_report(RESULTS_DIRECTORY, environment, fixtures, replay_records, scp_results)
    println("\nWrote ", joinpath(RESULTS_DIRECTORY, "report.md"))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
