using BenchmarkTools
using JuMP
using JuliaQOCO
using SparseArrays

include(joinpath(@__DIR__, "..", "examples", "scp_jump_reuse.jl"))

function fresh_jump_solve(horizon::Int, nx::Int, nu::Int; direct::Bool)
    case = build_jump_scp_case(; horizon, nx, nu, direct)
    optimize!(case.model)
    return objective_value(case.model)
end

function forced_rebuild_resolve!(case::JuMPSCPCase)
    MOI.set(
        backend(case.model),
        MOI.RawOptimizerAttribute("reuse_solver"),
        false,
    )
    optimize!(case.model)
    return nothing
end

function structural_rebuild(horizon::Int, nx::Int, nu::Int)
    case = build_jump_scp_case(; horizon, nx, nu)
    optimize!(case.model)
    @variable(case.model, structural_extra >= 0)
    set_objective_coefficient(case.model, structural_extra, 1.0)
    optimize!(case.model)
    return MOI.get(
        backend(case.model),
        MOI.RawOptimizerAttribute("rebuild_count"),
    )
end

function native_lower_bound!(solver::JuliaQOCO.Solver, index::Int, value::Float64)
    JuliaQOCO.update_c_entries!(solver, [index], [value])
    JuliaQOCO.solve!(solver)
    return nothing
end

function print_trial(label::AbstractString, trial)
    estimate = median(trial)
    println(
        "$label time_ms=$(estimate.time / 1e6) " *
        "alloc_kib=$(estimate.memory / 1024) allocations=$(estimate.allocs)",
    )
    return nothing
end

function run_jump_benchmarks(;
    horizon::Int = 24,
    nx::Int = 4,
    nu::Int = 2,
    samples::Int = 20,
)
    # Warm compilation paths.
    warm = build_jump_scp_case(; horizon = 4, nx = 2, nu = 1)
    optimize!(warm.model)
    update_jump_scp!(warm)
    optimize!(warm.model)

    construction_trial = @benchmark build_jump_scp_case(
        horizon = $horizon,
        nx = $nx,
        nu = $nu,
    ) samples = samples evals = 1
    fresh_cached_model_trial = @benchmark fresh_jump_solve(
        $horizon,
        $nx,
        $nu;
        direct = false,
    ) samples = samples evals = 1
    fresh_direct_trial = @benchmark fresh_jump_solve(
        $horizon,
        $nx,
        $nu;
        direct = true,
    ) samples = samples evals = 1

    nochange = build_jump_scp_case(; horizon, nx, nu)
    optimize!(nochange.model)
    optimize!(nochange.model)
    nochange_trial =
        @benchmark optimize!($(nochange.model)) samples = samples evals = 1

    vector_case = build_jump_scp_case(; horizon, nx, nu)
    optimize!(vector_case.model)
    vector_trial = @benchmark begin
        update_jump_scp_vectors!($vector_case)
        optimize!($(vector_case.model))
    end samples = samples evals = 1

    full_case = build_jump_scp_case(; horizon, nx, nu)
    optimize!(full_case.model)
    full_trial = @benchmark begin
        update_jump_scp!($full_case)
        optimize!($(full_case.model))
    end samples = samples evals = 1

    sparse_case = build_jump_scp_case(; horizon, nx, nu)
    optimize!(sparse_case.model)
    sparse_trial = @benchmark begin
        update_jump_scp!($sparse_case; matrix_fraction = 0.1)
        optimize!($(sparse_case.model))
    end samples = samples evals = 1

    rebuild_case = build_jump_scp_case(;
        horizon,
        nx,
        nu,
        reuse_solver = false,
    )
    optimize!(rebuild_case.model)
    forced_rebuild_trial = @benchmark forced_rebuild_resolve!($rebuild_case) samples =
        samples evals = 1
    structural_trial = @benchmark structural_rebuild($horizon, $nx, $nu) samples =
        min(samples, 5) evals = 1

    native_solver = MOI.get(backend(full_case.model), MOI.RawSolver())
    native_trial = @benchmark native_lower_bound!($native_solver, 1, -0.011) samples =
        samples evals = 1

    println("JuliaQOCO JuMP repeated-SCP benchmark")
    println("horizon=$horizon nx=$nx nu=$nu samples=$samples")
    print_trial("jump_construction", construction_trial)
    print_trial("fresh_jump_model_build_and_solve", fresh_cached_model_trial)
    print_trial("fresh_direct_model_build_and_solve", fresh_direct_trial)
    print_trial("cached_nochange_optimize", nochange_trial)
    print_trial("cached_vector_updates_and_solve", vector_trial)
    print_trial("cached_full_matrix_updates_and_solve", full_trial)
    print_trial("cached_sparse_matrix_updates_and_solve", sparse_trial)
    print_trial("forced_native_rebuild_nochange", forced_rebuild_trial)
    print_trial("structural_change_and_rebuild", structural_trial)
    print_trial("native_indexed_update_and_solve", native_trial)

    solver = MOI.get(backend(full_case.model), MOI.RawSolver())
    factor = solver.linsys.factor
    println(
        "native_setup_time_sec=$(solver.solution.setup_time_sec) " *
        "native_solve_time_sec=$(solver.solution.solve_time_sec) " *
        "ipm_iterations=$(solver.solution.iters) " *
        "kkt_nnz=$(nnz(factor.workspace.triuA)) factor_nnz=$(nnz(factor.L))",
    )
    println(
        "last_commit_time_sec=" *
        string(
            MOI.get(
                backend(full_case.model),
                MOI.RawOptimizerAttribute("last_commit_time_sec"),
            ),
        ),
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_jump_benchmarks()
end
