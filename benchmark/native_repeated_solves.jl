using Random
using SparseArrays
using Statistics

using JuliaQOCO

function make_box_qp_problem(n::Int)
    P = spdiagm(0 => fill(2.0, n))
    c = collect(range(-1.0, -2.0; length = n))
    A = sparse(ones(1, n))
    b = [1.0]
    G = spdiagm(0 => fill(-1.0, n))
    h = zeros(n)
    return P, c, A, b, G, h
end

function repeated_native_benchmark(; n::Int = 50, repeats::Int = 30, seed::Int = 1)
    rng = MersenneTwister(seed)
    P, c, A, b, G, h = make_box_qp_problem(n)
    settings = JuliaQOCO.Settings{Float64}(; verbose = false, profile = true)

    cold_wall = @elapsed begin
        solver = JuliaQOCO.Solver(P, c, A, b, G, h, n, Int[]; settings = settings)
        JuliaQOCO.solve!(solver)
    end
    cold_alloc = @allocated begin
        solver = JuliaQOCO.Solver(P, c, A, b, G, h, n, Int[]; settings = settings)
        JuliaQOCO.solve!(solver)
    end

    solver = JuliaQOCO.Solver(P, c, A, b, G, h, n, Int[]; settings = settings)
    JuliaQOCO.solve!(solver)
    ctrial = similar(c)
    @inbounds for i in eachindex(c)
        ctrial[i] = c[i] + 0.05 * sin(i)
    end
    JuliaQOCO.update_vector_data!(solver; c = ctrial)
    JuliaQOCO.solve!(solver)

    solve_times = Float64[]
    solve_allocs = Int[]
    iterations = Int[]
    predictor_times = Float64[]

    for _ in 1:repeats
        @inbounds for i in eachindex(c)
            ctrial[i] = c[i] + 0.05 * randn(rng)
        end
        JuliaQOCO.update_vector_data!(solver; c = ctrial)

        push!(solve_allocs, @allocated JuliaQOCO.solve!(solver))
        push!(solve_times, solver.solution.solve_time_sec)
        push!(iterations, solver.solution.iters)
        push!(predictor_times, solver.solution.profile.predictor_time_sec)
    end

    println("JuliaQOCO native repeated-solve benchmark")
    println("problem_size_n=$(n) repeats=$(repeats)")
    println("cold_wall_sec=$(cold_wall)")
    println("cold_alloc_bytes=$(cold_alloc)")
    println("warm_mean_solve_sec=$(mean(solve_times))")
    println("warm_median_solve_sec=$(median(solve_times))")
    println("warm_mean_alloc_bytes=$(mean(solve_allocs))")
    println("warm_median_alloc_bytes=$(median(solve_allocs))")
    println("warm_mean_iterations=$(mean(iterations))")
    println("warm_mean_predictor_sec=$(mean(predictor_times))")
    return nothing
end

function make_socp_problem(n::Int)
    q = n + 1
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for i in 1:n
        push!(rows, 1)
        push!(cols, i)
        push!(vals, -1 / sqrt(n))
        push!(rows, i + 1)
        push!(cols, i)
        push!(vals, -1.0)
    end
    P = spdiagm(0 => fill(1.0, n))
    c = fill(-0.1, n)
    G = sparse(rows, cols, vals, q, n)
    h = [1.5; zeros(q - 1)]
    return P, c, G, h, q
end

function repeated_socp_benchmark(; n::Int = 20, repeats::Int = 30, seed::Int = 1)
    rng = MersenneTwister(seed)
    P, c, G, h, q = make_socp_problem(n)
    settings = JuliaQOCO.Settings{Float64}(; verbose = false, profile = true)

    solver = JuliaQOCO.Solver(P, c, nothing, nothing, G, h, 0, [q]; settings = settings)
    JuliaQOCO.solve!(solver)

    solve_times = Float64[]
    solve_allocs = Int[]
    nt_times = Float64[]
    predictor_times = Float64[]

    ctrial = similar(c)
    @inbounds for i in eachindex(c)
        ctrial[i] = c[i] - 0.01 * cos(i)
    end
    JuliaQOCO.update_vector_data!(solver; c = ctrial)
    JuliaQOCO.solve!(solver)

    for k in 1:repeats
        @inbounds for i in eachindex(c)
            ctrial[i] = c[i] - 0.01 * sin(rand(rng) + i + k)
        end
        JuliaQOCO.update_vector_data!(solver; c = ctrial)
        push!(solve_allocs, @allocated JuliaQOCO.solve!(solver))
        push!(solve_times, solver.solution.solve_time_sec)
        push!(nt_times, solver.solution.profile.nt_scaling_time_sec + solver.solution.profile.nt_update_time_sec)
        push!(predictor_times, solver.solution.profile.predictor_time_sec)
    end

    println("JuliaQOCO repeated SOCP benchmark")
    println("problem_size_n=$(n) repeats=$(repeats)")
    println("warm_mean_solve_sec=$(mean(solve_times))")
    println("warm_median_solve_sec=$(median(solve_times))")
    println("warm_mean_alloc_bytes=$(mean(solve_allocs))")
    println("warm_median_alloc_bytes=$(median(solve_allocs))")
    println("warm_mean_nt_sec=$(mean(nt_times))")
    println("warm_mean_predictor_sec=$(mean(predictor_times))")
    return nothing
end

repeated_native_benchmark()
println()
repeated_socp_benchmark()