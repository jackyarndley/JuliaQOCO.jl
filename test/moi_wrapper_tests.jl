module MOIWrapperTests

using Test
using JuMP
using MathOptInterface

import JuliaQOCO

const MOI = MathOptInterface
const MOIU = MOI.Utilities

@testset "MOI Wrapper" begin
    @testset "Solver attributes" begin
        opt = JuliaQOCO.Optimizer()
        @test MOI.get(opt, MOI.SolverName()) == "JuliaQOCO"
        @test MOI.get(opt, MOI.SolverVersion()) == string(pkgversion(JuliaQOCO))
        @test MOI.is_empty(opt)
    end

    @testset "Settings" begin
        opt = JuliaQOCO.Optimizer(; max_iters = 100, abstol = 1e-6)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("max_iters")) == 100
        @test MOI.get(opt, MOI.RawOptimizerAttribute("abstol")) == 1e-6
        MOI.set(opt, MOI.RawOptimizerAttribute("reltol"), 1e-8)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("reltol")) == 1e-8
    end

    @testset "Silent" begin
        opt = JuliaQOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)
        @test MOI.get(opt, MOI.Silent()) == true
    end

    @testset "Simple QP via MOI" begin
        opt = JuliaQOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOIU.Model{Float64}()
        x = MOI.add_variables(model, 2)

        quad_terms = [
            MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(-1.0, x[1]),
            MOI.ScalarAffineTerm(-1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(opt, MOI.ResultCount()) == 1

        x1_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[1]])
        x2_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[2]])
        @test x1_val ≈ 0.5 atol = 1e-5
        @test x2_val ≈ 0.5 atol = 1e-5
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -0.5 atol = 1e-5
    end

    @testset "Linear objective with SOC via MOI" begin
        opt = JuliaQOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOIU.Model{Float64}()
        x = MOI.add_variables(model, 3)

        obj = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, x[1])],
            0.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        soc_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, x[3])),
            ],
            [0.0, 0.0, 0.0],
        )
        MOI.add_constraint(model, soc_func, MOI.SecondOrderCone(3))

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[3])),
            ],
            [-1.0, 0.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        x1_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[1]])
        x2_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[2]])
        x3_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[3]])
        @test x1_val ≈ 1.0 atol = 1e-4
        @test x2_val ≈ 1.0 atol = 1e-4
        @test abs(x3_val) < 1e-4
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol = 1e-4
    end

    @testset "Mixed cone ordering via MOI" begin
        opt = JuliaQOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOIU.Model{Float64}()
        x = MOI.add_variables(model, 4)

        obj = MOI.ScalarAffineFunction(
            [
                MOI.ScalarAffineTerm(1.0, x[1]),
                MOI.ScalarAffineTerm(1.0, x[4]),
            ],
            0.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        soc_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, x[3])),
            ],
            [0.0, 0.0, 0.0],
        )
        MOI.add_constraint(model, soc_func, MOI.SecondOrderCone(3))

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[3])),
            ],
            [-1.0, 0.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(2))

        nn_func = MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[4]))],
            [0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(1))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.VariablePrimal(), idxmap[x[1]]) ≈ 1.0 atol = 1e-4
        @test MOI.get(opt, MOI.VariablePrimal(), idxmap[x[4]]) ≈ 0.0 atol = 1e-4
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol = 1e-4
    end

    @testset "Maximization" begin
        opt = JuliaQOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOIU.Model{Float64}()
        x = MOI.add_variables(model, 2)

        quad_terms = [
            MOI.ScalarQuadraticTerm(-2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(-2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(1.0, x[1]),
            MOI.ScalarAffineTerm(1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.5 atol = 1e-4
    end

    @testset "Direct MOI coefficient updates" begin
        opt = JuliaQOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        @test MOI.supports_incremental_interface(opt)

        x = MOI.add_variables(opt, 2)
        quad_terms = [
            MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(-1.0, x[1]),
            MOI.ScalarAffineTerm(-1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(opt, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        eq_ci = MOI.add_constraint(opt, eq_func, MOI.Zeros(1))

        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(opt, nn_func, MOI.Nonnegatives(2))

        MOI.optimize!(opt)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.VariablePrimal(), x[1]) ≈ 0.5 atol = 1e-5
        @test MOI.get(opt, MOI.VariablePrimal(), x[2]) ≈ 0.5 atol = 1e-5

        raw_solver = MOI.get(opt, MOI.RawSolver())
        MOI.modify(opt, eq_ci, MOI.MultirowChange(x[2], [(1, 2.0)]))
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.VariablePrimal(), x[1]) ≈ 0.4 atol = 1e-5
        @test MOI.get(opt, MOI.VariablePrimal(), x[2]) ≈ 0.3 atol = 1e-5
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -0.45 atol = 1e-5
        @test MOI.get(opt, MOI.RawSolver()) === raw_solver

        MOI.modify(opt, eq_ci, MOI.VectorConstantChange([-1.2]))
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.VariablePrimal(), x[1]) ≈ 0.44 atol = 1e-5
        @test MOI.get(opt, MOI.VariablePrimal(), x[2]) ≈ 0.38 atol = 1e-5
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -0.482 atol = 1e-5
        @test MOI.get(opt, MOI.RawSolver()) === raw_solver
    end

    @testset "Empty model" begin
        opt = JuliaQOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)
        model = MOIU.Model{Float64}()
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.copy_to(opt, model)
        MOI.optimize!(opt)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.0
    end

    @testset "Iteration limit" begin
        opt = JuliaQOCO.Optimizer(; max_iters = 1)
        MOI.set(opt, MOI.Silent(), true)

        model = MOIU.Model{Float64}()
        x = MOI.add_variables(model, 2)

        quad_terms = [
            MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(-1.0, x[1]),
            MOI.ScalarAffineTerm(-1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        status = MOI.get(opt, MOI.TerminationStatus())
        @test status in (MOI.ITERATION_LIMIT, MOI.ALMOST_OPTIMAL)
    end

    @testset "Infeasible problem" begin
        opt = JuliaQOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOIU.Model{Float64}()
        x = MOI.add_variables(model, 1)

        obj = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, x[1])],
            0.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[1])),
            ],
            [-1.0, -2.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(2))

        MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        status = MOI.get(opt, MOI.TerminationStatus())
        @test status in (
            MOI.ITERATION_LIMIT,
            MOI.NUMERICAL_ERROR,
            MOI.ALMOST_OPTIMAL,
            MOI.OTHER_ERROR,
        )
    end
end

@testset "MOI.Test" begin
    optimizer = MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            JuliaQOCO.Optimizer(),
        ),
        Float64,
    )
    MOI.set(optimizer, MOI.Silent(), true)

    MOI.Test.runtests(
        optimizer,
        MOI.Test.Config(
            atol = 1e-3,
            rtol = 1e-3,
            optimal_status = MOI.OPTIMAL,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ObjectiveBound,
            ],
        );
        exclude = Any[
            "test_conic_ExponentialCone",
            "test_conic_DualExponentialCone",
            "test_conic_PowerCone",
            "test_conic_DualPowerCone",
            "test_conic_GeometricMeanCone",
            r"test_conic_RootDetConeTriangle.*",
            r"test_conic_RootDetConeSquare.*",
            r"test_conic_LogDetConeTriangle.*",
            r"test_conic_LogDetConeSquare.*",
            r"test_conic_PositiveSemidefiniteConeTriangle.*",
            r"test_conic_PositiveSemidefiniteConeSquare.*",
            "test_conic_RelativeEntropyCone",
            r"test_conic_NormSpectralCone.*",
            r"test_conic_NormNuclearCone.*",
            "test_conic_NormInfinityCone",
            r"test_conic_NormOneCone.*",
            "test_conic_HermitianPositiveSemidefiniteConeTriangle",
            "test_conic_NormCone",
            "test_basic_VectorOfVariables_",
            "test_solve_SOS",
            "test_constraint_ZeroOne",
            "test_constraint_Integer",
            "test_model_",
            r"test_conic_linear_INFEASIBLE.*",
            "test_conic_NonnegToNonworking",
            "test_conic_SecondOrderCone_INFEASIBLE",
            "test_conic_SecondOrderCone_negative_post_bound_2",
            "test_conic_SecondOrderCone_negative_post_bound_3",
            "test_conic_SecondOrderCone_no_initial_bound",
            r"test_conic_RotatedSecondOrderCone_INFEASIBLE.*",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            "test_infeasible",
            r"test_linear_INFEASIBLE.*",
            r"test_linear_DUAL_INFEASIBLE.*",
            "test_unbounded",
            "test_attribute_TimeLimitSec",
            "test_modification_",
            "test_objective_ObjectiveFunction_",
            "test_variable_",
            "test_constraint_",
            "test_solve_result_index",
            "test_quadratic_constraint_",
            "test_linear_integration_delete_variables",
            "test_solve_optimize_twice",
        ],
        exclude_tests_after = pkgversion(MOI),
    )
end

@testset "JuMP smoke test" begin
    model = JuMP.Model(JuliaQOCO.Optimizer)
    JuMP.set_silent(model)

    @variable(model, x[1:2] >= 0)
    @constraint(model, x[1] + x[2] == 1)
    @objective(model, Min, x[1]^2 + x[2]^2 - x[1] - x[2])

    JuMP.optimize!(model)

    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 0.5 atol = 1e-4
    @test JuMP.value(x[2]) ≈ 0.5 atol = 1e-4
    @test JuMP.objective_value(model) ≈ -0.5 atol = 1e-4
end

@testset "JuMP structural edits after optimize" begin
    model = JuMP.Model(JuliaQOCO.Optimizer)
    JuMP.set_silent(model)

    @variable(model, x[1:2] >= 0)
    @constraint(model, x[1] + x[2] == 1)
    @objective(model, Min, x[1]^2 + x[2]^2 - x[1] - x[2])

    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL

    @constraint(model, x[1] <= 0.4)
    JuMP.optimize!(model)

    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 0.4 atol = 1e-4
    @test JuMP.value(x[2]) ≈ 0.6 atol = 1e-4
    @test JuMP.objective_value(model) ≈ -0.48 atol = 1e-4
end

@testset "JuMP direct normalized coefficient updates" begin
    backend = MOI.Bridges.full_bridge_optimizer(JuliaQOCO.Optimizer(), Float64)
    model = JuMP.direct_model(backend)
    JuMP.set_silent(model)

    @variable(model, x[1:2])
    @constraint(model, x in MOI.Nonnegatives(2))
    con = @constraint(model, x[1] + x[2] == 1)
    @objective(model, Min, x[1]^2 + x[2]^2 - x[1] - x[2])

    JuMP.optimize!(model)

    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 0.5 atol = 1e-4
    @test JuMP.value(x[2]) ≈ 0.5 atol = 1e-4

    JuMP.set_normalized_coefficient(con, x[2], 2.0)
    JuMP.optimize!(model)

    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 0.4 atol = 1e-4
    @test JuMP.value(x[2]) ≈ 0.3 atol = 1e-4
    @test JuMP.objective_value(model) ≈ -0.45 atol = 1e-4
end

@testset "JuMP normalized rhs updates" begin
    model = JuMP.Model(JuliaQOCO.Optimizer)
    JuMP.set_silent(model)

    @variable(model, x[1:2] >= 0)
    con = @constraint(model, x[1] + x[2] == 1)
    @objective(model, Min, x[1]^2 + x[2]^2 - x[1] - x[2])

    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL

    JuMP.set_normalized_rhs(con, 1.2)
    JuMP.optimize!(model)

    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 0.6 atol = 1e-4
    @test JuMP.value(x[2]) ≈ 0.6 atol = 1e-4
    @test JuMP.objective_value(model) ≈ -0.48 atol = 1e-4
end

end
