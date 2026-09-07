# Compare the current solver against the audited revision it started from.
#
#   julia --project=benchmark benchmark/baseline_comparison.jl [revision]
#
# The audited revision is checked out into a temporary git worktree and driven
# by `baseline_worker.jl` in a separate process, because two versions of one
# package cannot be loaded into a single session. Both sides replay the same
# fixtures and are graded by the same independent oracle, so the comparison is
# accuracy-matched: a baseline run that fails the quality gate produces no
# speedup ratio at all.

using JSON
using Printf
using Statistics

const AUDITED_REVISION = "c2725866335a51a54b10450726c22c05c4127dc9"
const ROOT = dirname(@__DIR__)

include(joinpath(@__DIR__, "replay.jl"))
using .Replay
using .Replay.Fixtures

function baseline_results(revision::AbstractString)
    worktree = mktempdir()
    output = joinpath(worktree, "baseline.json")
    try
        run(`git -C $ROOT worktree add --detach $worktree $revision`)
        environment = joinpath(worktree, "benchmark-env")
        mkpath(environment)
        # A throwaway environment that develops the audited checkout.
        run(`julia --project=$environment -e "using Pkg; Pkg.develop(path=raw\"$worktree\")"`)
        run(`julia --project=$environment $(joinpath(@__DIR__, "baseline_worker.jl")) $(joinpath(@__DIR__, "fixtures.jl")) $(joinpath(ROOT, "test", "oracle.jl")) $output`)
        return JSON.parsefile(output)
    finally
        try
            run(`git -C $ROOT worktree remove --force $worktree`)
        catch error
            @warn "could not remove the temporary worktree" worktree error
        end
    end
end

function current_results(fixtures)
    records = run_replay(;
        fixtures = fixtures,
        repetitions = 3,
        include_references = false,
        verbose = false,
        solvers = ["juliaqoco-native-reuse"],
        workloads = ["full_update"],
    )
    return filter(r -> r.workload == "full_update", records)
end

function main(arguments::Vector{String} = ARGS)
    revision = isempty(arguments) ? AUDITED_REVISION : arguments[1]
    fixtures = [
        build_fixture(; name = "small", horizon = 10, nx = 4, nu = 2, steps = 6),
        build_fixture(; name = "medium", horizon = 30, nx = 6, nu = 3, steps = 6),
        build_fixture(; name = "large-soc", horizon = 40, nx = 6, nu = 3, steps = 3),
    ]
    println("Running the audited baseline at ", revision, " in a separate process")
    baseline = baseline_results(revision)
    println("Running the current solver")
    current = current_results(fixtures)

    lines = String[]
    push!(lines, "# Baseline comparison\n")
    push!(lines, "Audited revision: `$revision`.\n")
    push!(lines, "Workload: the full update-and-solve replay sequence. Both sides are graded")
    push!(lines, "by the same independent oracle at a relative tolerance of 1e-5. A ratio is")
    push!(lines, "reported only where both sides passed the gate on every replayed step.\n")
    push!(lines, "| fixture | baseline median (ms) | baseline pass/fail | current median (ms) | current pass/fail | accuracy-matched speedup |")
    push!(lines, "| --- | ---: | :---: | ---: | :---: | ---: |")
    for record in baseline
        name = record["fixture"]
        match = findfirst(r -> r.fixture == name, current)
        match === nothing && continue
        now = current[match]
        base_time = record["median_time_sec"]
        base_ok = record["quality_failures"] == 0 && base_time !== nothing
        now_ok = now.quality_failures == 0 && isfinite(now.median_time_sec)
        ratio = base_ok && now_ok ? @sprintf("%.2fx", base_time / now.median_time_sec) :
                "not comparable (a side failed the quality gate)"
        push!(lines, @sprintf(
            "| %s | %s | %d/%d | %s | %d/%d | %s |",
            name,
            base_time === nothing ? "n/a" : @sprintf("%.4g", 1e3 * base_time),
            record["quality_passes"], record["quality_failures"],
            isfinite(now.median_time_sec) ? @sprintf("%.4g", 1e3 * now.median_time_sec) : "n/a",
            now.quality_passes, now.quality_failures,
            ratio,
        ))
    end
    push!(lines, "")
    push!(lines, "Worst relative residual seen, baseline versus current:\n")
    for record in baseline
        name = record["fixture"]
        match = findfirst(r -> r.fixture == name, current)
        match === nothing && continue
        push!(lines, @sprintf(
            "- `%s`: baseline %.3e, current %.3e",
            name, record["worst_relative_residual"], current[match].worst_relative_residual,
        ))
    end

    report = join(lines, "\n") * "\n"
    directory = joinpath(@__DIR__, "results")
    mkpath(directory)
    write(joinpath(directory, "baseline_comparison.md"), report)
    print(report)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
