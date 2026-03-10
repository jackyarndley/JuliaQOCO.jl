module MOIWrapperTests

using Test
using JuMP
using MathOptInterface

import JuliaQOCO

const MOI = MathOptInterface
const MOIU = MOI.Utilities

@testset "JuMP smoke test" begin
    model = JuMP.Model(JuliaQOCO.Optimizer)
    JuMP.set_silent(model)
    @variable(model, x)
    @constraint(model, [x, 1.0] in SecondOrderCone())
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test isapprox(JuMP.value(x), 1.0; atol = 1e-6, rtol = 1e-6)
end

@testset "MOI standard tests" begin
    optimizer = JuliaQOCO.Optimizer{Float64}()
    MOI.set(optimizer, MOI.Silent(), true)
    bridged = MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()), optimizer),
        Float64,
    )

    config = MOI.Test.Config(
        atol = 1e-4,
        rtol = 1e-4,
        optimal_status = MOI.OPTIMAL,
        exclude = Any[
            MOI.VariableName,
            MOI.ConstraintName,
            MOI.VariableBasisStatus,
            MOI.ConstraintBasisStatus,
            MOI.ObjectiveBound,
            MOI.DualObjectiveValue,
            MOI.delete,
        ],
    )

    for testfn in (
        MOI.Test.test_linear_integration,
        MOI.Test.test_linear_integration_2,
        MOI.Test.test_linear_LessThan_and_GreaterThan,
        MOI.Test.test_linear_VectorAffineFunction,
        MOI.Test.test_quadratic_integration,
        MOI.Test.test_quadratic_duplicate_terms,
        MOI.Test.test_quadratic_nonhomogeneous,
        MOI.Test.test_conic_linear_VectorOfVariables,
        MOI.Test.test_conic_linear_VectorAffineFunction,
        MOI.Test.test_conic_SecondOrderCone_VectorOfVariables,
        MOI.Test.test_conic_SecondOrderCone_VectorAffineFunction,
        MOI.Test.test_conic_SecondOrderCone_Nonnegatives,
    )
        MOI.empty!(bridged)
        @testset "$(nameof(testfn))" begin
            testfn(bridged, config)
        end
    end
end

end
