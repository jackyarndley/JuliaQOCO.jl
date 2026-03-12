using BenchmarkTools
using LinearAlgebra
using Random
using SparseArrays

using JuliaQOCO

mutable struct SCPBenchmarkCase
    base_c::Vector{Float64}
    base_b::Vector{Float64}
    base_h::Vector{Float64}
    base_Ax::Vector{Float64}
    base_Gx::Vector{Float64}
    ctrial::Vector{Float64}
    btrial::Vector{Float64}
    htrial::Vector{Float64}
    Axtrial::Vector{Float64}
    Gxtrial::Vector{Float64}
    h_heads::Vector{Int}
    iter::Int
end

@inline function state_index(k::Int, i::Int, nx::Int)
    return k * nx + i
end

@inline function control_index(k::Int, i::Int, nx::Int, horizon::Int, nu::Int)
    return (horizon + 1) * nx + k * nu + i
end

@inline function trust_index(k::Int, nx::Int, horizon::Int, nu::Int)
    return (horizon + 1) * nx + horizon * nu + k
end

function build_scp_like_problem(; horizon::Int = 32, nx::Int = 6, nu::Int = 3)
    n = (horizon + 1) * nx + horizon * nu + horizon
    p = (horizon + 1) * nx
    trust_q = nx + nu + 1
    control_q = nu + 1
    qdims = vcat(fill(trust_q, horizon), fill(control_q, horizon))
    h_heads = Vector{Int}(undef, 2 * horizon)

    Pdiag = zeros(Float64, n)
    c = zeros(Float64, n)
    for k in 0:(horizon - 1)
        for i in 1:nu
            idx = control_index(k, i, nx, horizon, nu)
            Pdiag[idx] = 0.1 + 0.02 * i
            c[idx] = -0.01 * (1 + mod(k + i, 5))
        end
        tidx = trust_index(k + 1, nx, horizon, nu)
        Pdiag[tidx] = 1.0
        c[tidx] = 0.05
    end
    for i in 1:nx
        c[state_index(horizon, i, nx)] = -0.02
    end
    P = spdiagm(0 => Pdiag)

    arows = Int[]
    acols = Int[]
    avals = Float64[]
    b = zeros(Float64, p)

    x0 = collect(range(0.05, 0.05 * nx; length = nx))
    b[1:nx] .= x0
    for i in 1:nx
        push!(arows, i)
        push!(acols, state_index(0, i, nx))
        push!(avals, 1.0)
    end

    row_offset = nx
    for k in 0:(horizon - 1)
        for row in 1:nx
            eqrow = row_offset + row
            push!(arows, eqrow)
            push!(acols, state_index(k + 1, row, nx))
            push!(avals, 1.0)

            for col in 1:nx
                push!(arows, eqrow)
                push!(acols, state_index(k, col, nx))
                if row == col
                    push!(avals, -0.92 - 0.01 * sin(0.2 * (k + row)))
                else
                    push!(avals, -0.015 * cos(0.1 * (3 * row + col + k)))
                end
            end

            for col in 1:nu
                push!(arows, eqrow)
                push!(acols, control_index(k, col, nx, horizon, nu))
                push!(avals, -0.04 * cos(0.3 * (row + col + k)))
            end

            b[eqrow] = 0.02 * sin(0.15 * (k * nx + row))
        end
        row_offset += nx
    end
    A = sparse(arows, acols, avals, p, n)

    grows = Int[]
    gcols = Int[]
    gvals = Float64[]
    h = zeros(Float64, sum(qdims))
    cone_row = 1
    head_id = 1
    for k in 0:(horizon - 1)
        trust_row = cone_row
        h_heads[head_id] = trust_row
        head_id += 1
        h[trust_row] = 1.5 + 0.03 * (1 + mod(k, 4))
        push!(grows, trust_row)
        push!(gcols, trust_index(k + 1, nx, horizon, nu))
        push!(gvals, -1.0)
        for i in 1:nx
            push!(grows, trust_row + i)
            push!(gcols, state_index(k + 1, i, nx))
            push!(gvals, 0.20 + 0.01 * cos(0.2 * (k + i)))
        end
        for i in 1:nu
            push!(grows, trust_row + nx + i)
            push!(gcols, control_index(k, i, nx, horizon, nu))
            push!(gvals, 0.35 + 0.02 * sin(0.3 * (k + i)))
        end
        cone_row += trust_q

        control_row = cone_row
        h_heads[head_id] = control_row
        head_id += 1
        h[control_row] = 0.8 + 0.01 * (1 + mod(k, 3))
        for i in 1:nu
            push!(grows, control_row + i)
            push!(gcols, control_index(k, i, nx, horizon, nu))
            push!(gvals, 1.0)
        end
        cone_row += control_q
    end
    G = sparse(grows, gcols, gvals, sum(qdims), n)

    return P, c, A, b, G, h, qdims, h_heads
end

function make_benchmark_case(; horizon::Int = 32, nx::Int = 6, nu::Int = 3)
    P, c, A, b, G, h, qdims, h_heads = build_scp_like_problem(; horizon = horizon, nx = nx, nu = nu)
    case = SCPBenchmarkCase(
        copy(c),
        copy(b),
        copy(h),
        copy(A.nzval),
        copy(G.nzval),
        similar(c),
        similar(b),
        similar(h),
        similar(A.nzval),
        similar(G.nzval),
        h_heads,
        0,
    )
    copyto!(case.ctrial, case.base_c)
    copyto!(case.btrial, case.base_b)
    copyto!(case.htrial, case.base_h)
    copyto!(case.Axtrial, case.base_Ax)
    copyto!(case.Gxtrial, case.base_Gx)
    return P, c, A, b, G, h, qdims, case
end

function advance_case!(case::SCPBenchmarkCase)
    case.iter += 1
    phase = 0.13 * case.iter

    @inbounds for i in eachindex(case.ctrial, case.base_c)
        case.ctrial[i] = case.base_c[i] + 0.002 * sin(phase + 0.01 * i)
    end
    @inbounds for i in eachindex(case.btrial, case.base_b)
        case.btrial[i] = case.base_b[i] + 0.001 * cos(phase + 0.03 * i)
    end

    copyto!(case.htrial, case.base_h)
    @inbounds for (j, head) in pairs(case.h_heads)
        case.htrial[head] = case.base_h[head] + 0.01 * sin(phase + 0.2 * j)
    end

    @inbounds for i in eachindex(case.Axtrial, case.base_Ax)
        case.Axtrial[i] = case.base_Ax[i] * (1.0 + 0.004 * sin(phase + 0.005 * i))
    end
    @inbounds for i in eachindex(case.Gxtrial, case.base_Gx)
        case.Gxtrial[i] = case.base_Gx[i] * (1.0 + 0.003 * cos(phase + 0.007 * i))
    end

    return case
end

function prepare_cached_solver!(solver::JuliaQOCO.Solver, case::SCPBenchmarkCase)
    JuliaQOCO.solve!(solver)
    advance_case!(case)
    JuliaQOCO.update_vector_data!(solver; c = case.ctrial, b = case.btrial, h = case.htrial)
    JuliaQOCO.update_matrix_data!(solver; Ax = case.Axtrial, Gx = case.Gxtrial)
    JuliaQOCO.solve!(solver)
    return solver
end

@inline to_ms(ns::Real) = ns / 1.0e6
@inline to_kib(bytes::Real) = bytes / 1024.0

function print_estimate(label::AbstractString, estimate)
    println(
        label,
        ": time_ms=",
        round(to_ms(estimate.time); digits = 3),
        " gc_ms=",
        round(to_ms(estimate.gctime); digits = 3),
        " alloc_kib=",
        round(to_kib(estimate.memory); digits = 3),
        " allocs=",
        estimate.allocs,
    )
    return nothing
end

function print_trial_summary(label::AbstractString, trial::BenchmarkTools.Trial)
    print_estimate(label * "_min", minimum(trial))
    print_estimate(label * "_median", median(trial))
    return nothing
end

function precompile_solver_paths!()
    P, c, A, b, G, h, qdims, case = make_benchmark_case(; horizon = 8, nx = 4, nu = 2)
    settings = JuliaQOCO.Settings{Float64}(; verbose = false, profile = false)
    solver = JuliaQOCO.Solver(P, c, A, b, G, h, 0, qdims; settings = settings)
    prepare_cached_solver!(solver, case)
    return nothing
end

function run_profiled_fresh_solve(; horizon::Int = 384, nx::Int = 32, nu::Int = 12)
    P, c, A, b, G, h, qdims, _ = make_benchmark_case(; horizon = horizon, nx = nx, nu = nu)
    settings = JuliaQOCO.Settings{Float64}(; verbose = false, profile = true)
    solver = JuliaQOCO.Solver(P, c, A, b, G, h, 0, qdims; settings = settings)
    JuliaQOCO.solve!(solver)
    profile = solver.solution.profile
    println("profile_fresh_setup_sec=$(solver.solution.setup_time_sec)")
    println("profile_fresh_solve_sec=$(solver.solution.solve_time_sec)")
    println("profile_fresh_total_sec=$(solver.solution.setup_time_sec + solver.solution.solve_time_sec)")
    println("profile_fresh_iterations=$(solver.solution.iters)")
    println("profile_fresh_nt_sec=$(profile.nt_scaling_time_sec + profile.nt_update_time_sec)")
    println("profile_fresh_predictor_sec=$(profile.predictor_time_sec)")
    println("profile_fresh_stopping_sec=$(profile.stopping_time_sec)")
    println("profile_fresh_linsys_solves=$(profile.linsys_solves)")
    println("profile_fresh_nt_refactors=$(profile.nt_refactors)")
    return nothing
end

function run_profiled_cached_resolve(; horizon::Int = 384, nx::Int = 32, nu::Int = 12)
    P, c, A, b, G, h, qdims, case = make_benchmark_case(; horizon = horizon, nx = nx, nu = nu)
    settings = JuliaQOCO.Settings{Float64}(; verbose = false, profile = true)
    solver = JuliaQOCO.Solver(P, c, A, b, G, h, 0, qdims; settings = settings)
    prepare_cached_solver!(solver, case)
    advance_case!(case)
    JuliaQOCO.update_vector_data!(solver; c = case.ctrial, b = case.btrial, h = case.htrial)
    JuliaQOCO.update_matrix_data!(solver; Ax = case.Axtrial, Gx = case.Gxtrial)
    JuliaQOCO.solve!(solver)
    profile = solver.solution.profile
    println("profile_cached_resolve_sec=$(solver.solution.solve_time_sec)")
    println("profile_cached_iterations=$(solver.solution.iters)")
    println("profile_cached_nt_sec=$(profile.nt_scaling_time_sec + profile.nt_update_time_sec)")
    println("profile_cached_predictor_sec=$(profile.predictor_time_sec)")
    println("profile_cached_stopping_sec=$(profile.stopping_time_sec)")
    println("profile_cached_linsys_solves=$(profile.linsys_solves)")
    println("profile_cached_nt_refactors=$(profile.nt_refactors)")
    return nothing
end

function run_benchmarks(; horizon::Int = 384, nx::Int = 32, nu::Int = 12, samples::Int = 10)
    println("JuliaQOCO SCP-like benchmark")
    println("horizon=$(horizon) nx=$(nx) nu=$(nu)")
    precompile_solver_paths!()
    run_profiled_fresh_solve(; horizon = horizon, nx = nx, nu = nu)
    run_profiled_cached_resolve(; horizon = horizon, nx = nx, nu = nu)

    settings = JuliaQOCO.Settings{Float64}(; verbose = false, profile = false)

    P, c, A, b, G, h, qdims, _ = make_benchmark_case(; horizon = horizon, nx = nx, nu = nu)
    fresh_trial = @benchmark begin
        solver = JuliaQOCO.Solver($P, $c, $A, $b, $G, $h, 0, $qdims; settings = $settings)
        JuliaQOCO.solve!(solver)
    end evals = 1 samples = samples

    P, c, A, b, G, h, qdims, vector_case = make_benchmark_case(; horizon = horizon, nx = nx, nu = nu)
    vector_solver = JuliaQOCO.Solver(P, c, A, b, G, h, 0, qdims; settings = settings)
    prepare_cached_solver!(vector_solver, vector_case)
    vector_trial = @benchmark JuliaQOCO.update_vector_data!($vector_solver; c = $vector_case.ctrial, b = $vector_case.btrial, h = $vector_case.htrial) setup = (advance_case!($vector_case)) evals = 1 samples = samples

    P, c, A, b, G, h, qdims, matrix_case = make_benchmark_case(; horizon = horizon, nx = nx, nu = nu)
    matrix_solver = JuliaQOCO.Solver(P, c, A, b, G, h, 0, qdims; settings = settings)
    prepare_cached_solver!(matrix_solver, matrix_case)
    matrix_trial = @benchmark JuliaQOCO.update_matrix_data!($matrix_solver; Ax = $matrix_case.Axtrial, Gx = $matrix_case.Gxtrial) setup = (advance_case!($matrix_case)) evals = 1 samples = samples

    P, c, A, b, G, h, qdims, solve_case = make_benchmark_case(; horizon = horizon, nx = nx, nu = nu)
    solve_solver = JuliaQOCO.Solver(P, c, A, b, G, h, 0, qdims; settings = settings)
    prepare_cached_solver!(solve_solver, solve_case)
    solve_trial = @benchmark begin
        JuliaQOCO.update_vector_data!($solve_solver; c = $solve_case.ctrial, b = $solve_case.btrial, h = $solve_case.htrial)
        JuliaQOCO.update_matrix_data!($solve_solver; Ax = $solve_case.Axtrial, Gx = $solve_case.Gxtrial)
        JuliaQOCO.solve!($solve_solver)
    end setup = (advance_case!($solve_case)) evals = 1 samples = samples

    print_trial_summary("compiled_fresh_setup_and_solve", fresh_trial)
    print_trial_summary("cached_vector_update", vector_trial)
    print_trial_summary("cached_matrix_update", matrix_trial)
    print_trial_summary("cached_update_and_resolve", solve_trial)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
