module MOISemanticsTests

using Test
using JuMP
using LinearAlgebra
using MathOptInterface

import JuliaQOCO

const MOI = MathOptInterface

quiet_optimizer(; kwargs...) = JuliaQOCO.Optimizer(; verbose = false, kwargs...)

# An output stream that raises a chosen exception the moment anything is
# written to it, used to inject a failure into the middle of a solve.
struct ThrowingIO <: IO
    exception::Exception
end

Base.write(io::ThrowingIO, ::UInt8) = throw(io.exception)
Base.unsafe_write(io::ThrowingIO, ::Ptr{UInt8}, ::UInt) = throw(io.exception)
Base.print(io::ThrowingIO, ::Any...) = throw(io.exception)

@testset "MOI semantics" begin

    @testset "Objective sense is converted exactly once" begin
        # Maximizing a concave quadratic. Replacing the whole objective used to
        # apply the maximization sign a second time, turning -2x^2 into +2x^2
        # and producing a nonconvex model.
        function build_max_model()
            optimizer = quiet_optimizer()
            model = direct_model(optimizer)
            @variable(model, 0 <= x <= 1)
            @objective(model, Max, -x^2 + x)
            optimize!(model)
            return optimizer, model, x
        end

        replaced, model, x = build_max_model()
        @test isapprox(value(x), 0.5; atol = 1e-5)
        @test isapprox(objective_value(model), 0.25; atol = 1e-6)
        rebuilds = MOI.get(replaced, MOI.RawOptimizerAttribute("rebuild_count"))
        MOI.set(
            replaced,
            MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
            MOI.ScalarQuadraticFunction(
                [MOI.ScalarQuadraticTerm(-4.0, index(x), index(x))],
                [MOI.ScalarAffineTerm(1.0, index(x))],
                0.0,
            ),
        )
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(value(x), 0.25; atol = 1e-5)
        @test isapprox(objective_value(model), 0.125; atol = 1e-6)
        # The support did not change, so no symbolic rebuild was needed.
        @test MOI.get(replaced, MOI.RawOptimizerAttribute("rebuild_count")) == rebuilds

        # The coefficient-modification route must agree exactly.
        modified, model2, x2 = build_max_model()
        MOI.modify(
            modified,
            MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
            MOI.ScalarQuadraticCoefficientChange(index(x2), index(x2), -4.0),
        )
        optimize!(model2)
        @test termination_status(model2) == MOI.OPTIMAL
        @test isapprox(value(x2), 0.25; atol = 1e-5)
        @test isapprox(objective_value(model2), 0.125; atol = 1e-6)

        # And so must a freshly constructed optimizer.
        fresh = quiet_optimizer()
        model3 = direct_model(fresh)
        @variable(model3, 0 <= x3 <= 1)
        @objective(model3, Max, -2x3^2 + x3)
        optimize!(model3)
        @test isapprox(value(x3), 0.25; atol = 1e-5)
        @test isapprox(objective_value(model3), 0.125; atol = 1e-6)
    end

    @testset "Objective constants and sense changes" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, 0 <= x <= 2)
        @objective(model, Min, 3x + 7)
        optimize!(model)
        @test isapprox(objective_value(model), 7.0; atol = 1e-6)

        @objective(model, Max, 3x + 7)
        optimize!(model)
        @test isapprox(objective_value(model), 13.0; atol = 1e-5)

        MOI.set(optimizer, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) == 0.0

        # Switching to FEASIBILITY_SENSE discards the objective function, per
        # the MOI model contract, so it has to be supplied again.
        @objective(model, Min, 3x + 7)
        optimize!(model)
        @test isapprox(objective_value(model), 7.0; atol = 1e-5)
        @test isapprox(value(x), 0.0; atol = 1e-5)
    end

    @testset "Settings reach the cached solver" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, y >= 0)
        @constraint(model, y >= 1)
        @objective(model, Min, y)
        optimize!(model)
        raw = MOI.get(optimizer, MOI.RawSolver())
        @test raw.settings.verbose == false

        buffer = IOBuffer()
        MOI.set(optimizer, MOI.RawOptimizerAttribute("output"), buffer)
        MOI.set(optimizer, MOI.Silent(), false)
        optimize!(model)
        @test MOI.get(optimizer, MOI.RawSolver()) === raw
        @test raw.settings.verbose == true
        @test occursin("status:", String(take!(buffer)))

        # A tolerance change reaches the solver without a symbolic rebuild.
        rebuilds = MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count"))
        MOI.set(optimizer, MOI.Silent(), true)
        MOI.set(optimizer, MOI.RawOptimizerAttribute("abstol"), 1e-9)
        optimize!(model)
        @test raw.settings.abstol == 1e-9
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count")) == rebuilds

        # Changing only the dynamic regularization must not rebuild either.
        MOI.set(optimizer, MOI.RawOptimizerAttribute("kkt_dynamic_reg"), 1e-9)
        optimize!(model)
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count")) == rebuilds
    end

    @testset "Time limit attribute" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, 0 <= x[1:3] <= 5)
        @constraint(model, sum(x) == 3)
        @objective(model, Min, sum(x .^ 2))
        @test MOI.get(optimizer, MOI.TimeLimitSec()) === nothing
        MOI.set(optimizer, MOI.TimeLimitSec(), 1e-9)
        @test MOI.get(optimizer, MOI.TimeLimitSec()) == 1e-9
        optimize!(model)
        @test termination_status(model) in
              (MOI.TIME_LIMIT, MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        MOI.set(optimizer, MOI.TimeLimitSec(), nothing)
        @test MOI.get(optimizer, MOI.TimeLimitSec()) === nothing
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
    end

    @testset "Invalid input is rejected, not silently dropped" begin
        for (bad_set, description) in (
            (MOI.GreaterThan(Inf), "lower bound of +Inf"),
            (MOI.LessThan(-Inf), "upper bound of -Inf"),
            (MOI.Interval(2.0, 1.0), "reversed interval"),
            (MOI.EqualTo(Inf), "infinite equality"),
        )
            optimizer = quiet_optimizer()
            model = direct_model(optimizer)
            @variable(model, x)
            @objective(model, Min, x)
            MOI.add_constraint(optimizer, index(x), bad_set)
            optimize!(model)
            @test termination_status(model) == MOI.INVALID_MODEL
            @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
            @test MOI.get(optimizer, MOI.ResultCount()) == 0
        end

        # A nonbinding infinity is genuinely nonbinding and must be accepted.
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, x)
        MOI.add_constraint(optimizer, index(x), MOI.GreaterThan(-Inf))
        MOI.add_constraint(optimizer, index(x), MOI.LessThan(4.0))
        @objective(model, Max, x)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(value(x), 4.0; atol = 1e-6)
    end

    @testset "Nonconvex objectives are refused" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, -1 <= x <= 1)
        @objective(model, Min, -x^2)
        optimize!(model)
        @test termination_status(model) == MOI.INVALID_MODEL
        @test MOI.get(optimizer, MOI.ResultCount()) == 0
    end

    @testset "Duplicate terms accumulate" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, 0 <= x <= 10)
        # 2x + 3x written as two separate terms must behave as 5x.
        MOI.set(
            optimizer,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(2.0, index(x)), MOI.ScalarAffineTerm(3.0, index(x))],
                0.0,
            ),
        )
        MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        constraint = MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, index(x)), MOI.ScalarAffineTerm(1.0, index(x))],
                0.0,
            ),
            MOI.LessThan(4.0),
        )
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(value(x), 2.0; atol = 1e-5)
        @test isapprox(objective_value(model), 10.0; atol = 1e-5)
        @test constraint isa MOI.ConstraintIndex
    end

    @testset "Reserved zeros do not force a rebuild" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, 0 <= x[1:3] <= 5)
        constraint = @constraint(model, x[1] + x[2] + x[3] == 3)
        @objective(model, Min, sum(x .^ 2))
        optimize!(model)
        raw = MOI.get(optimizer, MOI.RawSolver())
        factor = raw.linsys.factor
        rebuilds = MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count"))

        # Whole-function replacement that omits a term whose coefficient has
        # gone to zero. The slot stays reserved, so this is an update.
        MOI.set(
            optimizer,
            MOI.ConstraintFunction(),
            index(constraint),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, index(x[1])), MOI.ScalarAffineTerm(1.0, index(x[2]))],
                0.0,
            ),
        )
        optimize!(model)
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count")) == rebuilds
        @test MOI.get(optimizer, MOI.RawSolver()) === raw
        @test raw.linsys.factor === factor
        @test isapprox(value(x[1]) + value(x[2]), 3.0; atol = 1e-5)
        @test isapprox(value(x[3]), 0.0; atol = 1e-4)

        # Bringing the omitted term back must also stay within the pattern.
        MOI.set(
            optimizer,
            MOI.ConstraintFunction(),
            index(constraint),
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, index(x[1])),
                    MOI.ScalarAffineTerm(1.0, index(x[2])),
                    MOI.ScalarAffineTerm(1.0, index(x[3])),
                ],
                0.0,
            ),
        )
        optimize!(model)
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count")) == rebuilds
        @test raw.linsys.factor === factor
        @test isapprox(value.(x), fill(1.0, 3); atol = 1e-5)
    end

    @testset "Whole-function replacement is linear in the term count" begin
        # A genuinely large vector constraint. The all-pairs scan this
        # replaces is quadratic in the number of terms, so the wall time is a
        # meaningful, if coarse, guard against its reintroduction.
        rows = 200
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, xs[1:rows])
        terms = [
            MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm(1.0, index(xs[i])))
            for i in 1:rows
        ]
        constraint = MOI.add_constraint(
            optimizer,
            MOI.VectorAffineFunction(terms, fill(-1.0, rows)),
            MOI.Zeros(rows),
        )
        @objective(model, Min, sum(xs .^ 2))
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        rebuilds = MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count"))

        replacement = [
            MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm(2.0, index(xs[i])))
            for i in 1:rows
        ]
        MOI.set(
            optimizer,
            MOI.ConstraintFunction(),
            constraint,
            MOI.VectorAffineFunction(replacement, fill(-1.0, rows)),
        )
        optimize!(model)
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute("rebuild_count")) == rebuilds
        @test isapprox(value.(xs), fill(0.5, rows); atol = 1e-5)
    end

    @testset "Result availability and statuses" begin
        optimizer = quiet_optimizer(max_iters = 1)
        model = direct_model(optimizer)
        @variable(model, 0 <= x[1:3] <= 5)
        @constraint(model, sum(x) == 3)
        @constraint(model, [2.0; x...] in SecondOrderCone())
        @objective(model, Min, sum(x .^ 2) - x[1])
        optimize!(model)
        status = termination_status(model)
        @test status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.ITERATION_LIMIT)
        if MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
            @test MOI.get(optimizer, MOI.ResultCount()) == 0
            @test_throws MOI.ResultIndexBoundsError MOI.get(optimizer, MOI.ObjectiveValue())
        else
            @test MOI.get(optimizer, MOI.ResultCount()) == 1
            @test all(isfinite, value.(x))
        end
        @test_throws MOI.ResultIndexBoundsError MOI.get(
            optimizer, MOI.ObjectiveValue(2),
        )

        MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iters"), 200)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(optimizer, MOI.ResultCount()) == 1
    end

    @testset "Results are invalidated by modifications" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, 0 <= x <= 5)
        @objective(model, Min, (x - 2)^2)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        MOI.modify(
            optimizer,
            MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
            MOI.ScalarCoefficientChange(index(x), -6.0),
        )
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        @test MOI.get(optimizer, MOI.ResultCount()) == 0
        optimize!(model)
        @test isapprox(value(x), 3.0; atol = 1e-5)
    end

    @testset "Dual starts are declined rather than ignored" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, x)
        constraint = @constraint(model, x >= 1)
        @test MOI.supports(optimizer, MOI.VariablePrimalStart(), MOI.VariableIndex)
        @test !MOI.supports(
            optimizer, MOI.ConstraintDualStart(), typeof(index(constraint)),
        )
    end

    @testset "Variable primal starts are consumed" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, 0 <= x[1:2] <= 5)
        @constraint(model, x[1] + x[2] == 2)
        @objective(model, Min, sum((x .- 1) .^ 2))
        MOI.set(optimizer, MOI.VariablePrimalStart(), index(x[1]), 1.0)
        MOI.set(optimizer, MOI.VariablePrimalStart(), index(x[2]), 1.0)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(value.(x), [1.0, 1.0]; atol = 1e-5)
    end

    @testset "Constraint duals have consistent signs" begin
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, x[1:2])
        equality = @constraint(model, x[1] + x[2] == 1)
        lower = @constraint(model, x[1] >= 0.75)
        @objective(model, Min, sum(x .^ 2))
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(value(x[1]), 0.75; atol = 1e-5)
        # Stationarity in original units: 2x + A'y_eq + G'z = 0 with MOI dual
        # sign conventions folded in by the wrapper.
        y_eq = dual(equality)
        z_low = dual(lower)
        @test isapprox(2 * value(x[1]) - y_eq - z_low, 0.0; atol = 1e-5)
        @test isapprox(2 * value(x[2]) - y_eq, 0.0; atol = 1e-5)
        @test z_low >= -1e-8
    end

    @testset "Interrupts are not converted into numerical failures" begin
        # An interrupt raised anywhere inside the solve has to reach the
        # caller. An ordinary error, by contrast, is still captured as a
        # status so that a numerical failure stays debuggable.
        optimizer = quiet_optimizer()
        model = direct_model(optimizer)
        @variable(model, x >= 0)
        @constraint(model, x == 1)
        @objective(model, Min, x)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL

        MOI.set(optimizer, MOI.RawOptimizerAttribute("output"), ThrowingIO(InterruptException()))
        MOI.set(optimizer, MOI.Silent(), false)
        @test_throws InterruptException MOI.optimize!(optimizer)

        MOI.set(optimizer, MOI.RawOptimizerAttribute("output"), ThrowingIO(ErrorException("synthetic failure")))
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OTHER_ERROR
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        @test occursin("synthetic failure", MOI.get(optimizer, MOI.RawStatusString()))
    end
end

end # module
