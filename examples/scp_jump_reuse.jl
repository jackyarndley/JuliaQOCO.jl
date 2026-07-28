using JuMP
using JuliaQOCO

const MOI = JuMP.MOI

mutable struct JuMPSCPCase
    model::JuMP.Model
    x
    u
    trust
    dynamics
    horizon::Int
    nx::Int
    nu::Int
    iteration::Int
end

function build_jump_scp_case(;
    horizon::Int = 12,
    nx::Int = 3,
    nu::Int = 2,
    direct::Bool = true,
    reuse_solver::Bool = true,
)
    optimizer = JuliaQOCO.Optimizer(;
        verbose = false,
        reuse_solver,
        scaling_mode = :once,
        warm_start_mode = :primal_dual,
    )
    model = direct ? JuMP.direct_model(optimizer) :
            JuMP.Model(() -> optimizer)
    @variable(model, x[0:horizon, 1:nx])
    @variable(model, -1 <= u[0:(horizon - 1), 1:nu] <= 1)
    @variable(model, 0 <= trust[1:horizon] <= 1.5)

    for i in 1:nx
        @constraint(model, x[0, i] == 0.05 * i)
    end
    dynamics = Matrix{JuMP.ConstraintRef}(undef, horizon, nx)
    for k in 0:(horizon - 1), i in 1:nx
        dynamics[k + 1, i] = @constraint(
            model,
            x[k + 1, i] -
            sum(
                (
                    i == j ? 0.82 + 0.01 * i :
                    0.01 + 0.001 * (i + j)
                ) * x[k, j] for j in 1:nx
            ) -
            sum((0.04 + 0.003 * i + 0.002 * j) * u[k, j] for j in 1:nu) ==
            0.002 * sin(k + i),
        )
    end
    for k in 1:horizon
        @constraint(model, [trust[k]; [x[k, i] for i in 1:nx]] in SecondOrderCone())
        @constraint(
            model,
            [1.0; [u[k - 1, j] for j in 1:nu]] in SecondOrderCone(),
        )
    end
    @objective(
        model,
        Min,
        sum(0.1 * u[k, j]^2 for k in 0:(horizon - 1), j in 1:nu) +
        sum(0.05 * trust[k] for k in 1:horizon) -
        sum(0.01 * x[horizon, i] for i in 1:nx),
    )
    return JuMPSCPCase(
        model,
        x,
        u,
        trust,
        dynamics,
        horizon,
        nx,
        nu,
        0,
    )
end

function update_jump_scp!(case::JuMPSCPCase; matrix_fraction::Float64 = 1.0)
    case.iteration += 1
    phase = 0.17 * case.iteration
    matrix_stride = matrix_fraction >= 1 ? 1 :
                    max(1, round(Int, inv(matrix_fraction)))
    for k in 0:(case.horizon - 1), i in 1:case.nx
        constraint = case.dynamics[k + 1, i]
        set_normalized_rhs(
            constraint,
            0.002 * sin(k + i + phase) + 0.0005 * cos(phase + i),
        )
        for j in 1:case.nx
            if (j - 1) % matrix_stride == 0
                base =
                    i == j ? 0.82 + 0.01 * i :
                    0.01 + 0.001 * (i + j)
                coefficient =
                    -base *
                    (1 + 0.01 * sin(phase + 0.1 * (k + i + j)))
                set_normalized_coefficient(
                    constraint,
                    case.x[k, j],
                    coefficient,
                )
            end
        end
        for j in 1:case.nu
            if (case.nx + j - 1) % matrix_stride == 0
                coefficient =
                    -(0.04 + 0.003 * i + 0.002 * j) *
                    (1 + 0.02 * cos(phase + 0.2 * (k + i + j)))
                set_normalized_coefficient(
                    constraint,
                    case.u[k, j],
                    coefficient,
                )
            end
        end
    end
    for k in 1:case.horizon
        set_upper_bound(case.trust[k], 1.2 + 0.1 * sin(phase + 0.1 * k))
        set_objective_coefficient(
            case.model,
            case.trust[k],
            0.05 + 0.005 * cos(phase + 0.2 * k),
        )
    end
    return case
end

function update_jump_scp_vectors!(case::JuMPSCPCase)
    case.iteration += 1
    phase = 0.17 * case.iteration
    for k in 0:(case.horizon - 1), i in 1:case.nx
        set_normalized_rhs(
            case.dynamics[k + 1, i],
            0.002 * sin(k + i + phase),
        )
    end
    for k in 1:case.horizon
        set_upper_bound(case.trust[k], 1.2 + 0.1 * sin(phase + 0.1 * k))
        set_objective_coefficient(
            case.model,
            case.trust[k],
            0.05 + 0.005 * cos(phase + 0.2 * k),
        )
    end
    return case
end

function run_jump_scp_example(; iterations::Int = 5)
    construction = @timed build_jump_scp_case()
    case = construction.value
    first_solve = @timed optimize!(case.model)
    solver = MOI.get(backend(case.model), MOI.RawSolver())
    println(
        "construction_time_sec=$(construction.time) " *
        "construction_alloc_bytes=$(construction.bytes)",
    )
    println(
        "initial_optimize_time_sec=$(first_solve.time) " *
        "initial_optimize_alloc_bytes=$(first_solve.bytes) " *
        "native_setup_time_sec=$(solver.solution.setup_time_sec)",
    )
    for iteration in 1:iterations
        update_measurement = @timed update_jump_scp!(case)
        solve_measurement = @timed optimize!(case.model)
        reused = MOI.get(backend(case.model), MOI.RawSolver()) === solver
        println(
            "iteration=$iteration update_time_sec=$(update_measurement.time) " *
            "update_alloc_bytes=$(update_measurement.bytes) " *
            "optimize_time_sec=$(solve_measurement.time) " *
            "optimize_alloc_bytes=$(solve_measurement.bytes) " *
            "native_solve_time_sec=$(solver.solution.solve_time_sec) " *
            "ipm_iterations=$(solver.solution.iters) solver_reused=$reused",
        )
    end
    return case
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_jump_scp_example()
end
