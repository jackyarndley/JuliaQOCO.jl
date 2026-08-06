using Test
using Random
using SparseArrays
using LinearAlgebra
using JuliaQOCO

quiet_settings(; kwargs...) = JuliaQOCO.Settings{Float64}(
    ; verbose = false, scaling_mode = :once, ruiz_iters = 3, kwargs...
)

@testset "Cone kernels" begin
    u = [2.0, 0.2, 0.1]
    v = [1.5, 0.3, -0.2]
    product = similar(u)
    JuliaQOCO.cone_product!(product, u, v, 0, [3])
    @test product ≈ [2.0 * 1.5 + 0.2 * 0.3 + 0.1 * -0.2, 2.0 * 0.3 + 1.5 * 0.2, 2.0 * -0.2 + 1.5 * 0.1]
    @test JuliaQOCO.cone_residual(u, 0, [3]) < 0
end

@testset "Validation" begin
    @test_throws ArgumentError JuliaQOCO.CoreSolver(
        spzeros(2, 2), [1.0], nothing, nothing, nothing, nothing, 0, Int[];
        settings = quiet_settings(),
    )
    @test_throws ArgumentError JuliaQOCO.CoreSolver(
        sparse([1.0 2.0; 2.0 1.0]), [0.0, 0.0], nothing, nothing, nothing, nothing, 0, Int[];
        settings = quiet_settings(),
    )
    @test_throws ArgumentError JuliaQOCO.validate_settings(
        JuliaQOCO.Settings{Float64}(; scaling_mode = :invalid),
    )
end

@testset "Compact NT scaling" begin
    solver = JuliaQOCO.CoreSolver(
        spzeros(1, 1), [0.0], nothing, nothing,
        sparse([1], [1], [-1.0], 3, 1), [1.0, 1.0, 1.0], 0, [3];
        settings = quiet_settings(),
    )
    solver.work.s .= [1.2, 0.1, -0.2]
    solver.work.z .= [1.1, -0.1, 0.15]
    JuliaQOCO.compute_nt_scaling!(solver)
    q = 3
    dense_W = zeros(3, 3)
    dense_Winv = zeros(3, 3)
    for j in 1:q
        e = zeros(3)
        e[j] = 1.0
        JuliaQOCO.nt_multiply_W!(view(dense_W, :, j), e, solver.data, solver.work)
        JuliaQOCO.nt_multiply_Winv!(view(dense_Winv, :, j), e, solver.data, solver.work)
    end
    @test dense_W * dense_Winv ≈ Matrix{Float64}(I, 3, 3) atol = 1e-10 rtol = 1e-10
    expected = [dense_W' * dense_W][1]
    offset = solver.work.Wtri_offsets[1]
    @test solver.work.WtW[offset:(offset + 5)] ≈ [expected[j, k] for j in 1:q for k in 1:j]
end

@testset "Convex QP and repeated updates" begin
    P = sparse([1, 2], [1, 2], [2.0, 2.0], 2, 2)
    c = [-1.0, -1.0]
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2)
    b = [1.0]
    G = sparse([1, 2], [1, 2], [-1.0, -1.0], 2, 2)
    h = [0.0, 0.0]
    settings = quiet_settings()
    solver = JuliaQOCO.CoreSolver(P, c, A, b, G, h, 2, Int[]; settings)
    JuliaQOCO._solve!(solver)
    @test solver.solution.status == JuliaQOCO.QOCO_SOLVED
    @test solver.solution.x ≈ [0.5, 0.5] atol = 1e-6
    scaling = (copy(solver.scaling.Druiz), copy(solver.scaling.Eruiz), copy(solver.scaling.Fruiz))
    JuliaQOCO.update_c_entries!(solver, [1, 2], [-2.0, -1.0])
    JuliaQOCO.update_b_entries!(solver, [1], [1.0])
    JuliaQOCO._solve!(solver)
    @test solver.solution.status == JuliaQOCO.QOCO_SOLVED
    @test solver.solution.x ≈ [0.75, 0.25] atol = 1e-5
    @test solver.scaling.Druiz == scaling[1]
    @test solver.scaling.Eruiz == scaling[2]
    @test solver.scaling.Fruiz == scaling[3]
end

@testset "Warm starts and best iterate" begin
    solver = JuliaQOCO.CoreSolver(
        sparse([1, 2], [1, 2], [2.0, 2.0], 2, 2), [-1.0, -1.0],
        sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2), [1.0],
        sparse([1, 2], [1, 2], [-1.0, -1.0], 2, 2), [0.0, 0.0], 2, Int[];
        settings = quiet_settings(max_iters = 1),
    )
    JuliaQOCO._solve!(solver)
    @test solver.solution.best_valid
    @test all(isfinite, solver.solution.x)
    JuliaQOCO.warm_start!(solver; x = [0.5, 0.5], s = [0.5, 0.5], y = [0.0], z = [1.0, 1.0])
    solver.settings.max_iters = 100
    JuliaQOCO._solve!(solver)
    @test solver.solution.status in (JuliaQOCO.QOCO_SOLVED, JuliaQOCO.QOCO_SOLVED_INACCURATE)
end
