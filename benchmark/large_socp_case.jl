using SparseArrays

using JuliaQOCO

function make_large_block_socp_problem(block_count::Int, block_dim::Int)
    vars_per_block = block_dim
    cone_dim = block_dim + 1
    n = block_count * vars_per_block
    qdims = fill(cone_dim, block_count)

    P = spdiagm(0 => fill(0.5, n))
    c = zeros(n)
    grows = Int[]
    gcols = Int[]
    gvals = Float64[]
    h = zeros(sum(qdims))

    row_offset = 0
    for block_id = 1:block_count
        var_offset = (block_id - 1) * vars_per_block
        first_row = row_offset + 1
        h[first_row] = 2.0
        for local_id = 1:vars_per_block
            idx = var_offset + local_id
            c[idx] = -0.01 * (1 + mod(local_id + block_id, 7))
            push!(grows, first_row)
            push!(gcols, idx)
            push!(gvals, -1 / sqrt(vars_per_block))
            push!(grows, first_row + local_id)
            push!(gcols, idx)
            push!(gvals, -1.0)
        end
        row_offset += cone_dim
    end
    G = sparse(grows, gcols, gvals, sum(qdims), n)

    arows = Int[]
    acols = Int[]
    avals = Float64[]
    b = zeros(block_count)
    b[1] = 1.0

    for local_id = 1:vars_per_block
        push!(arows, 1)
        push!(acols, local_id)
        push!(avals, 1.0)
    end
    for block_id = 2:block_count
        row_id = block_id
        prev_offset = (block_id - 2) * vars_per_block
        curr_offset = (block_id - 1) * vars_per_block
        for local_id = 1:vars_per_block
            push!(arows, row_id)
            push!(acols, prev_offset + local_id)
            push!(avals, -1 / vars_per_block)
            push!(arows, row_id)
            push!(acols, curr_offset + local_id)
            push!(avals, 1 / vars_per_block)
        end
    end
    A = sparse(arows, acols, avals, block_count, n)

    return P, c, A, b, G, h, qdims
end

function warm_solve_time(block_count::Int, block_dim::Int)
    P, c, A, b, G, h, qdims = make_large_block_socp_problem(block_count, block_dim)
    settings = JuliaQOCO.Settings{Float64}(; verbose = false, profile = true)
    solver = JuliaQOCO.Solver(P, c, A, b, G, h, 0, qdims; settings = settings)
    JuliaQOCO.solve!(solver)

    ctrial = copy(c)
    @inbounds for i in eachindex(ctrial)
        ctrial[i] += 0.001 * sin(i)
    end
    JuliaQOCO.update_vector_data!(solver; c = ctrial)
    JuliaQOCO.solve!(solver)

    return solver.solution.solve_time_sec, solver.solution.profile
end

function calibrate_large_case(; target_time_sec::Float64 = 1.0, block_dim::Int = 24)
    candidates = ((128, 64), (128, 96), (128, 128), (192, 128), (256, 128), (192, 160), (256, 160))
    best_block_count, best_block_dim = first(candidates)
    best_time = 0.0
    best_profile = nothing

    for (block_count, candidate_dim) in candidates
        solve_time, profile = warm_solve_time(block_count, candidate_dim)
        println(
            "blocks=$(block_count) block_dim=$(candidate_dim) warm_solve_sec=$(solve_time) predictor_sec=$(profile.predictor_time_sec) nt_sec=$(profile.nt_scaling_time_sec + profile.nt_update_time_sec)",
        )
        best_block_count = block_count
        best_block_dim = candidate_dim
        best_time = solve_time
        best_profile = profile
        solve_time >= target_time_sec && break
    end

    return best_block_count, best_block_dim, best_time, best_profile
end

if abspath(PROGRAM_FILE) == @__FILE__
    selected_blocks, selected_block_dim, selected_warm_solve_sec, selected_profile = calibrate_large_case()
    println()
    println("selected_blocks=$(selected_blocks)")
    println("selected_block_dim=$(selected_block_dim)")
    println("selected_warm_solve_sec=$(selected_warm_solve_sec)")
    println("selected_predictor_sec=$(selected_profile.predictor_time_sec)")
    println("selected_nt_sec=$(selected_profile.nt_scaling_time_sec + selected_profile.nt_update_time_sec)")
end