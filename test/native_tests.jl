module NativeTests

using Test
using SparseArrays

import JuliaQOCO

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
    @test_throws ArgumentError JuliaQOCO.solve(P, c; l = 0, q = Int[])

    G = sparse([1.0; 0.0;;])
    h = [1.0, 0.0]
    @test_throws ArgumentError JuliaQOCO.solve(spzeros(1, 1), [1.0]; G = G, h = h, l = 0, q = Int[])
end

@testset "Native solves" begin
    lp = JuliaQOCO.solve(spzeros(1, 1), [1.0]; G = sparse([-1.0;;]), h = [-1.0], l = 1, q = Int[])
    @test lp.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(lp.solution.x[1], 1.0; atol = 1e-7, rtol = 1e-7)

    qp = JuliaQOCO.solve(sparse([2.0;;]), [-4.0]; l = 0, q = Int[])
    @test qp.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(qp.solution.x[1], 2.0; atol = 1e-7, rtol = 1e-7)
    @test isapprox(qp.solution.obj, -4.0; atol = 1e-7, rtol = 1e-7)

    soc = JuliaQOCO.solve(spzeros(1, 1), [1.0]; G = sparse([-1.0; 0.0;;]), h = [0.0, 1.0], l = 0, q = [2])
    @test soc.solution.status == JuliaQOCO.QOCO_SOLVED
    @test isapprox(soc.solution.x[1], 1.0; atol = 1e-6, rtol = 1e-6)
end

end
