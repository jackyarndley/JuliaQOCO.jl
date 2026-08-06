module MOIWrapperTests

using Test
using JuMP
using MathOptInterface

import JuliaQOCO

const MOI = MathOptInterface

@testset "MOI interface" begin
    @testset "Only Optimizer is public" begin
        public_names = names(JuliaQOCO)
        @test :Optimizer in public_names
        @test :solve! ∉ public_names
        @test :solve ∉ public_names
        @test :Solver ∉ public_names
        @test :CoreSolver ∉ public_names
        @test :_solve! ∉ public_names
    end

    @testset "JuMP direct model and reuse" begin
        model = JuMP.direct_model(JuliaQOCO.Optimizer(
            verbose = false,
            scaling_mode = :once,
            warm_start_mode = :primal_dual,
        ))
        @variable(model, 0 <= x[1:2] <= 2)
        equality = @constraint(model, x[1] + x[2] == 1)
        soc = @constraint(model, [1.0, x[1], x[2]] in SecondOrderCone())
        @objective(model, Min, x[1]^2 + 2x[2]^2 - x[1])
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL

        optimizer = backend(model)
        raw_solver = MOI.get(optimizer, MOI.RawSolver())
        factor = raw_solver.linsys.factor
        rebuild_count = MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count"))

        # Whole-function replacements retain exactly the same support.
        MOI.set(
            optimizer,
            MOI.ConstraintFunction(),
            JuMP.index(equality),
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm{Float64}[
                    MOI.ScalarAffineTerm(1.5, JuMP.index(x[1])),
                    MOI.ScalarAffineTerm(1.0, JuMP.index(x[2])),
                ],
                0.0,
            ),
        )
        MOI.set(
            optimizer,
            MOI.ConstraintFunction(),
            JuMP.index(soc),
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, JuMP.index(x[1]))),
                    MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(0.5, JuMP.index(x[2]))),
                ],
                [1.2, 0.0, 0.0],
            ),
        )
        MOI.set(
            optimizer,
            MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
            MOI.ScalarQuadraticFunction(
                MOI.ScalarQuadraticTerm{Float64}[
                    MOI.ScalarQuadraticTerm(2.0, JuMP.index(x[1]), JuMP.index(x[1])),
                    MOI.ScalarQuadraticTerm(4.0, JuMP.index(x[2]), JuMP.index(x[2])),
                ],
                MOI.ScalarAffineTerm{Float64}[
                    MOI.ScalarAffineTerm(-1.5, JuMP.index(x[1])),
                ],
                0.0,
            ),
        )
        optimize!(model)
        @test termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        @test MOI.get(optimizer, MOI.RawSolver()) === raw_solver
        @test raw_solver.linsys.factor === factor
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count")) == rebuild_count
        @test all(isfinite, value.(x))
        @test_throws Exception MOI.get(optimizer, MOI.DualObjectiveValue())
    end

    @testset "Structural changes rebuild once" begin
        model = JuMP.direct_model(JuliaQOCO.Optimizer(verbose = false, scaling_mode = :once))
        @variable(model, x[1:2] >= 0)
        constraint = @constraint(model, x[1] + x[2] == 1)
        @objective(model, Min, x[1] + x[2])
        optimize!(model)
        optimizer = backend(model)
        count = MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count"))
        set_normalized_coefficient(constraint, x[1], 0.0)
        set_normalized_coefficient(constraint, x[1], 1.0)
        optimize!(model)
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count")) == count
        @variable(model, extra >= 0)
        set_objective_coefficient(model, extra, 1.0)
        optimize!(model)
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count")) == count + 1
    end

    @testset "Statuses and diagnostics" begin
        optimizer = JuliaQOCO.Optimizer(verbose = false, max_iters = 1)
        model = JuMP.direct_model(optimizer)
        @variable(model, x >= 0)
        @constraint(model, x == 1)
        @objective(model, Min, x)
        optimize!(model)
        @test termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.ITERATION_LIMIT)
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("structure_generation")) > 0
        @test MOI.get(optimizer, MOI.SolveTimeSec()) >= 0
    end

    @testset "Feasibility objective" begin
        optimizer = JuliaQOCO.Optimizer(verbose = false)
        model = JuMP.direct_model(optimizer)
        @variable(model, x)
        @constraint(model, x >= 1)
        @objective(model, Min, x)
        optimize!(model)
        MOI.set(optimizer, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) == 0.0
    end
end

end
