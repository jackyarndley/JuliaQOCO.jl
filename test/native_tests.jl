module NativeTests

using Test
using Random
using LinearAlgebra
using SparseArrays

import JuliaQOCO

quiet_settings() = JuliaQOCO.Settings{Float64}(; verbose = false)

@testset "Cone kernels" begin
    x = [6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [9.0, 10.0, 11.0, 6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p = similar(x)
    d = similar(x)

    JuliaQOCO.cone_product!(p, x, y, 3, [5, 5])
    JuliaQOCO.cone_division!(d, x, y, 3, [5, 5])

    @test isapprox(
        p,
        [54.0, 70.0, 88.0, 130.0, 19.0, 26.0, 33.0, 40.0, 130.0, 19.0, 26.0, 33.0, 40.0];
        atol = 1e-12,
        rtol = 1e-12,
    )
    @test isapprox(
        d,
        [
            1.5,
            10.0 / 7.0,
            1.375,
            2.2264150943396226,
            2.5471698113207544,
            1.3207547169811316,
            0.09433962264150907,
            -1.1320754716981127,
            2.2264150943396226,
            2.5471698113207544,
            1.3207547169811316,
            0.09433962264150907,
            -1.1320754716981127,
        ];
        atol = 1e-12,
        rtol = 1e-12,
    )
end

@testset "Validation" begin
    P = sparse([1.0 0.0; 0.0 1.0])
    c = [1.0]
    @test_throws ArgumentError JuliaQOCO.solve(P, c; l = 0, q = Int[], settings = quiet_settings())

    G = sparse([1.0; 0.0;;])
    h = [1.0, 0.0]
    @test_throws ArgumentError JuliaQOCO.solve(spzeros(1, 1), [1.0]; G = G, h = h, l = 0, q = Int[], settings = quiet_settings())
end

@testset "Verbosity" begin
    default_settings = JuliaQOCO.default_settings(Float64)
    @test default_settings.verbose
    @test default_settings.output === stdout

    verbose_io = IOBuffer()
    verbose_settings = JuliaQOCO.Settings{Float64}(; output = verbose_io)
    JuliaQOCO.solve(spzeros(1, 1), [1.0]; G = sparse([-1.0;;]), h = [-1.0], l = 1, q = Int[], settings = verbose_settings)
    verbose_output = String(take!(verbose_io))
    @test occursin("QOCO - Quadratic Objective Conic Optimizer", verbose_output)
    @test occursin("|  Iter  |", verbose_output)
    @test occursin("status:", verbose_output)

    quiet_io = IOBuffer()
    quiet_settings = JuliaQOCO.Settings{Float64}(; verbose = false, output = quiet_io)
    JuliaQOCO.solve(spzeros(1, 1), [1.0]; G = sparse([-1.0;;]), h = [-1.0], l = 1, q = Int[], settings = quiet_settings)
    quiet_output = String(take!(quiet_io))
    @test isempty(quiet_output)
end

@testset "Profiling" begin
    profiled = JuliaQOCO.solve(
        spzeros(1, 1),
        [1.0];
        G = sparse([-1.0;;]),
        h = [-1.0],
        l = 1,
        q = Int[],
        settings = JuliaQOCO.Settings{Float64}(; verbose = false, profile = true),
    )
    profile = profiled.solution.profile
    @test profile.problem_data_time_sec >= 0.0
    @test profile.workspace_time_sec >= 0.0
    @test profile.linsys_time_sec >= 0.0
    @test profile.initialize_time_sec >= 0.0
    @test profile.linsys_solves >= 1
    @test profile.nt_refactors >= 1
end

@testset "Native solves" begin
    lp = JuliaQOCO.solve(spzeros(1, 1), [1.0]; G = sparse([-1.0;;]), h = [-1.0], l = 1, q = Int[], settings = quiet_settings())
    @test lp.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(lp.solution.x[1], 1.0; atol = 1e-7, rtol = 1e-7)

    qp = JuliaQOCO.solve(sparse([2.0;;]), [-4.0]; l = 0, q = Int[], settings = quiet_settings())
    @test qp.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(qp.solution.x[1], 2.0; atol = 1e-7, rtol = 1e-7)
    @test isapprox(qp.solution.obj, -4.0; atol = 1e-7, rtol = 1e-7)

    soc = JuliaQOCO.solve(spzeros(1, 1), [1.0]; G = sparse([-1.0; 0.0;;]), h = [0.0, 1.0], l = 0, q = [2], settings = quiet_settings())
    @test soc.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(soc.solution.x[1], 1.0; atol = 1e-6, rtol = 1e-6)
end

@testset "SOC line search" begin
    solver = JuliaQOCO.Solver(spzeros(1, 1), [1.0], nothing, nothing, sparse([-1.0; 0.0;;]), [0.0, 1.0], 0, [2]; settings = quiet_settings())
    u = [2.0, 0.5]
    Du = [-1.1, 0.7]
    step = JuliaQOCO.linesearch!(solver, u, Du, 0.99)
    bisect_step = JuliaQOCO.bisection_search!(solver, u, Du, 0.99)
    trial = u .+ step .* Du
    @test JuliaQOCO.cone_residual(trial, 0, [2]) < 0.0
    @test step >= bisect_step

    rhs = [0.0, 0.0, -1.1, 0.7]
    step_from = JuliaQOCO.linesearch_from!(solver, u, rhs, 3, 0.99)
    bisect_from = JuliaQOCO.bisection_search_from!(solver, u, rhs, 3, 0.99)
    trial_from = u .+ step_from .* rhs[3:4]
    @test JuliaQOCO.cone_residual(trial_from, 0, [2]) < 0.0
    @test step_from >= bisect_from
end

@testset "Analytical SOC line search properties" begin
    rng = MersenneTwister(42)
    solver = JuliaQOCO.Solver(
        spzeros(1, 1),
        [0.0],
        nothing,
        nothing,
        sparse([0.0; 0.0; 0.0;;]),
        [1.0, 0.0, 0.0],
        0,
        [3];
        settings = quiet_settings(),
    )
    for _ in 1:250
        tail = randn(rng, 2)
        u = [norm(tail) + 0.1 + rand(rng); tail]
        direction = randn(rng, 3)
        step = JuliaQOCO.linesearch!(solver, u, direction, 0.99)
        trial = u .+ step .* direction
        @test JuliaQOCO.cone_residual(trial, 0, [3]) <= 1e-10

        lo = 0.0
        hi = 1.0
        if JuliaQOCO.cone_residual(u .+ hi .* direction, 0, [3]) >= 0
            for _ in 1:80
                mid = (lo + hi) / 2
                if JuliaQOCO.cone_residual(u .+ mid .* direction, 0, [3]) < 0
                    lo = mid
                else
                    hi = mid
                end
            end
        else
            lo = 1.0
        end
        @test step ≈ 0.99 * lo atol = 5e-11 rtol = 5e-10
    end
end

@testset "NT scaling consistency" begin
    P = spzeros(2, 2)
    c = [0.0, 0.0]
    G = sparse([1, 2], [1, 2], [-1.0, -1.0], 2, 2)
    h = [1.0, 0.5]
    solver = JuliaQOCO.Solver(P, c, nothing, nothing, G, h, 0, [2]; settings = quiet_settings())
    solver.work.s .= [2.0, 0.25]
    solver.work.z .= [1.5, 0.1]
    JuliaQOCO.compute_nt_scaling!(solver)
    q = solver.data.q[1]
    offset = solver.work.Wfull_offsets[1]
    tri_offset = solver.work.Wtri_offsets[1]
    Wblock = reshape(copy(@view solver.work.Wfull[offset:(offset + q * q - 1)]), q, q)
    gram = transpose(Wblock) * Wblock
    expected = [gram[j, k] for j in 1:q for k in 1:j]
    actual = solver.work.WtW[tri_offset:(tri_offset + length(expected) - 1)]
    @test isapprox(actual, expected; atol = 1e-10, rtol = 1e-10)

    rng = MersenneTwister(7)
    for _ in 1:100
        tail_s = 0.2 .* randn(rng, 1)
        tail_z = 0.2 .* randn(rng, 1)
        solver.work.s .= [norm(tail_s) + 0.5 + rand(rng); tail_s]
        solver.work.z .= [norm(tail_z) + 0.5 + rand(rng); tail_z]
        JuliaQOCO.compute_nt_scaling!(solver)
        input = randn(rng, 2)
        dense_W = similar(input)
        dense_Winv = similar(input)
        structured_W = similar(input)
        structured_Winv = similar(input)
        JuliaQOCO.nt_multiply!(
            dense_W,
            solver.work.Wfull,
            input,
            solver.data,
            solver.work,
        )
        JuliaQOCO.nt_multiply!(
            dense_Winv,
            solver.work.Winvfull,
            input,
            solver.data,
            solver.work,
        )
        JuliaQOCO.nt_multiply_W!(
            structured_W,
            input,
            solver.data,
            solver.work,
        )
        JuliaQOCO.nt_multiply_Winv!(
            structured_Winv,
            input,
            solver.data,
            solver.work,
        )
        @test structured_W ≈ dense_W atol = 1e-11 rtol = 1e-11
        @test structured_Winv ≈ dense_Winv atol = 1e-11 rtol = 1e-11
    end
end

@testset "Indexed updates and frozen scaling" begin
    P = sparse([1, 2], [1, 2], [2.0, 4.0], 2, 2)
    c = [-2.0, -1.0]
    A = sparse([1, 1], [1, 2], [1.0, 2.0], 1, 2)
    b = [1.0]
    G = sparse([1, 2], [1, 2], [-1.0, -1.0], 2, 2)
    h = [0.0, 0.0]
    settings = JuliaQOCO.Settings{Float64}(;
        verbose = false,
        ruiz_iters = 4,
        scaling_mode = :once,
    )
    solver = JuliaQOCO.Solver(P, c, A, b, G, h, 2, Int[]; settings)
    JuliaQOCO.solve!(solver)
    scaling_before = (
        copy(solver.scaling.Druiz),
        copy(solver.scaling.Eruiz),
        copy(solver.scaling.Fruiz),
        solver.scaling.k,
    )
    JuliaQOCO.update_P_entries!(solver, [1], [3.0])
    JuliaQOCO.update_A_entries!(solver, [2], [1.5])
    JuliaQOCO.update_G_entries!(solver, [1], [-0.8])
    JuliaQOCO.update_c_entries!(solver, [1], [-1.8])
    JuliaQOCO.update_b_entries!(solver, [1], [1.1])
    JuliaQOCO.update_h_entries!(solver, [2], [0.1])
    JuliaQOCO.solve!(solver)
    @test solver.solution.status == JuliaQOCO.QOCO_SOLVED
    @test solver.scaling.Druiz == scaling_before[1]
    @test solver.scaling.Eruiz == scaling_before[2]
    @test solver.scaling.Fruiz == scaling_before[3]
    @test solver.scaling.k == scaling_before[4]

    fresh = JuliaQOCO.solve(
        sparse([1, 2], [1, 2], [3.0, 4.0], 2, 2),
        [-1.8, -1.0];
        A = sparse([1, 1], [1, 2], [1.0, 1.5], 1, 2),
        b = [1.1],
        G = sparse([1, 2], [1, 2], [-0.8, -1.0], 2, 2),
        h = [0.0, 0.1],
        l = 2,
        q = Int[],
        settings = settings,
    )
    @test solver.solution.x ≈ fresh.solution.x atol = 2e-6 rtol = 2e-6
    @test solver.solution.obj ≈ fresh.solution.obj atol = 2e-6 rtol = 2e-6
end

@testset "Warm starts and in-place updates" begin
    P = sparse([1, 2], [1, 2], [2.0, 2.0], 2, 2)
    c = [-1.0, -1.0]
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2)
    b = [1.0]
    G = sparse([1, 2], [1, 2], [-1.0, -1.0], 2, 2)
    h = [0.0, 0.0]

    solver = JuliaQOCO.Solver(P, c, A, b, G, h, 2, Int[]; settings = quiet_settings())
    JuliaQOCO.solve!(solver)
    @test solver.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(solver.solution.x, [0.5, 0.5]; atol = 1e-6, rtol = 1e-6)

    JuliaQOCO.update_vector_data!(solver; c = [-2.0, -1.0])
    JuliaQOCO.solve!(solver)
    @test solver.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(solver.solution.x, [0.75, 0.25]; atol = 1e-5, rtol = 1e-5)

    JuliaQOCO.update_matrix_data!(solver; Ax = [1.0, 2.0])
    JuliaQOCO.solve!(solver)
    @test solver.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(solver.solution.x, [0.8, 0.1]; atol = 1e-5, rtol = 1e-5)
    @test isapprox(solver.solution.s, [0.8, 0.1]; atol = 1e-5, rtol = 1e-5)

    JuliaQOCO.warm_start!(solver; x = [0.8, 0.1], s = [0.8, 0.1], y = [0.6], z = [1.0, 1.0])
    JuliaQOCO.solve!(solver)
    @test solver.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(solver.solution.x, [0.8, 0.1]; atol = 1e-5, rtol = 1e-5)
end

@testset "Generated fixed-pattern KKT backend" begin
    P = sparse([1, 2], [1, 2], [2.0, 4.0], 2, 2)
    c = [-2.0, -1.0]
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2)
    b = [1.0]
    G = sparse([1, 2], [1, 2], [-1.0, -1.0], 2, 2)
    h = [0.0, 0.0]
    standard = JuliaQOCO.Solver(
        P,
        c,
        A,
        b,
        G,
        h,
        2,
        Int[];
        settings = JuliaQOCO.Settings{Float64}(;
            verbose = false,
            kkt_backend = :qdldl,
            scaling_mode = :once,
        ),
    )
    generated = JuliaQOCO.Solver(
        P,
        c,
        A,
        b,
        G,
        h,
        2,
        Int[];
        settings = JuliaQOCO.Settings{Float64}(;
            verbose = false,
            kkt_backend = :generated,
            scaling_mode = :once,
        ),
    )
    @test JuliaQOCO.active_kkt_backend(standard) == :qdldl
    @test JuliaQOCO.active_kkt_backend(generated) == :generated
    @test standard.linsys.factor.workspace.Lx ==
          generated.linsys.factor.workspace.Lx
    @test standard.linsys.factor.workspace.Dinv ==
          generated.linsys.factor.workspace.Dinv

    JuliaQOCO.solve!(standard)
    JuliaQOCO.solve!(generated)
    @test standard.solution.status == generated.solution.status ==
          JuliaQOCO.QOCO_SOLVED
    @test standard.solution.x == generated.solution.x
    @test standard.solution.obj == generated.solution.obj
    kkt_rhs = [0.2, -0.1, 0.3, -0.4, 0.5]
    standard_product = similar(kkt_rhs)
    generated_product = similar(kkt_rhs)
    JuliaQOCO.kkt_multiply!(
        standard_product,
        kkt_rhs,
        standard.data,
        standard.work,
        standard.linsys.factor.generated_pattern,
    )
    JuliaQOCO.kkt_multiply!(
        generated_product,
        kkt_rhs,
        generated.data,
        generated.work,
        generated.linsys.factor.generated_pattern,
    )
    @test standard_product == generated_product

    JuliaQOCO.update_vector_data!(
        standard;
        c = [-2.2, -0.8],
        b = [1.1],
    )
    JuliaQOCO.update_vector_data!(
        generated;
        c = [-2.2, -0.8],
        b = [1.1],
    )
    JuliaQOCO.update_matrix_data!(
        standard;
        Px = [2.2, 3.8],
        Ax = [1.0, 1.5],
        Gx = [-0.9, -1.1],
    )
    JuliaQOCO.update_matrix_data!(
        generated;
        Px = [2.2, 3.8],
        Ax = [1.0, 1.5],
        Gx = [-0.9, -1.1],
    )
    JuliaQOCO.solve!(standard)
    JuliaQOCO.solve!(generated)
    @test standard.solution.status == generated.solution.status ==
          JuliaQOCO.QOCO_SOLVED
    @test standard.solution.x == generated.solution.x
    @test standard.solution.obj == generated.solution.obj
    @test @allocated(JuliaQOCO.QDLDL.refactor!(generated.linsys.factor)) == 0
    @test all(iszero, generated.linsys.factor.workspace.factor_fwork)

    standard_soc = JuliaQOCO.solve(
        spzeros(1, 1),
        [1.0];
        G = sparse([-1.0; 0.0;;]),
        h = [0.0, 1.0],
        q = [2],
        settings = JuliaQOCO.Settings{Float64}(;
            verbose = false,
            kkt_backend = :qdldl,
        ),
    )
    generated_soc = JuliaQOCO.solve(
        spzeros(1, 1),
        [1.0];
        G = sparse([-1.0; 0.0;;]),
        h = [0.0, 1.0],
        q = [2],
        settings = JuliaQOCO.Settings{Float64}(;
            verbose = false,
            kkt_backend = :generated,
        ),
    )
    @test standard_soc.solution.status == generated_soc.solution.status ==
          JuliaQOCO.QOCO_SOLVED
    @test standard_soc.solution.x == generated_soc.solution.x
    @test standard_soc.solution.obj == generated_soc.solution.obj

    auto = JuliaQOCO.Solver(
        P,
        c,
        A,
        b,
        G,
        h,
        2,
        Int[];
        settings = JuliaQOCO.Settings{Float64}(;
            verbose = false,
            kkt_backend = :auto,
            generated_max_dimension = 1,
        ),
    )
    @test JuliaQOCO.active_kkt_backend(auto) == :qdldl
    auto_generated = JuliaQOCO.Solver(
        P,
        c,
        A,
        b,
        G,
        h,
        2,
        Int[];
        settings = JuliaQOCO.Settings{Float64}(;
            verbose = false,
            kkt_backend = :auto,
        ),
    )
    @test JuliaQOCO.active_kkt_backend(auto_generated) == :generated
    @test_throws ArgumentError JuliaQOCO.validate_settings(
        JuliaQOCO.Settings{Float64}(; kkt_backend = :unknown),
    )
end

end
