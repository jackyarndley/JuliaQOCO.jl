# A real sequential convex programming loop, used as the end-to-end benchmark.
#
# The problem is a nondimensional planar orbit transfer: a spacecraft under
# two-body gravity and bounded thrust has to reach a target position and
# velocity. Nothing about it is a framework - it is one dynamics function, one
# discretization, one linearization, and one outer loop with trust-region
# acceptance and a virtual-control penalty - but every ingredient that makes a
# convex subproblem sequence realistic is present:
#
#   * the dynamics are relinearized about the current reference every outer
#     iteration, so A and B genuinely change;
#   * steps are accepted or rejected on the *nonlinear* defect, so a faster
#     subproblem solve that produces a worse step is not rewarded;
#   * the trust radius shrinks on rejection and grows on acceptance;
#   * the virtual-control penalty weight changes between iterations.
#
# The final answer is verified against the nonlinear dynamics by independent
# re-propagation, so a run that converges quickly to a trajectory that does not
# fly cannot be reported as a win.

module NonlinearSCP

using JuMP
using LinearAlgebra
using Printf

export SCPResult, run_scp, scp_environment

const STATE_DIMENSION = 4
const CONTROL_DIMENSION = 2

# Two-body acceleration plus thrust, nondimensionalized so that the
# gravitational parameter is one.
function dynamics(state::AbstractVector, control::AbstractVector)
    r = @view state[1:2]
    v = @view state[3:4]
    distance = sqrt(r[1]^2 + r[2]^2)
    gravity = -r ./ distance^3
    return [v[1], v[2], gravity[1] + control[1], gravity[2] + control[2]]
end

function dynamics_jacobians(state::AbstractVector, control::AbstractVector)
    r = @view state[1:2]
    distance = sqrt(r[1]^2 + r[2]^2)
    d3 = distance^3
    d5 = distance^5
    dgdr = [
        -1/d3+3*r[1]*r[1]/d5   3*r[1]*r[2]/d5
        3*r[2]*r[1]/d5        -1/d3+3*r[2]*r[2]/d5
    ]
    A = zeros(STATE_DIMENSION, STATE_DIMENSION)
    A[1, 3] = 1.0
    A[2, 4] = 1.0
    A[3:4, 1:2] = dgdr
    B = zeros(STATE_DIMENSION, CONTROL_DIMENSION)
    B[3, 1] = 1.0
    B[4, 2] = 1.0
    return A, B
end

# One explicit Runge-Kutta 4 step. Used both for the discretization inside the
# linearization and, independently, for the final quality check.
function rk4_step(state::AbstractVector, control::AbstractVector, step::Float64)
    k1 = dynamics(state, control)
    k2 = dynamics(state .+ 0.5 * step .* k1, control)
    k3 = dynamics(state .+ 0.5 * step .* k2, control)
    k4 = dynamics(state .+ step .* k3, control)
    return state .+ (step / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

# Discrete linearization about the reference. A first-order expansion of the
# RK4 map is approximated by the matrix exponential-free Euler product, which
# is accurate to the order of the step and is the usual choice for this kind
# of subproblem.
function discrete_linearization(state::AbstractVector, control::AbstractVector, step::Float64)
    Ac, Bc = dynamics_jacobians(state, control)
    Ad = I + step * Ac + 0.5 * step^2 * Ac * Ac
    Bd = step * Bc + 0.5 * step^2 * Ac * Bc
    next = rk4_step(state, control, step)
    residual = next .- Ad * state .- Bd * control
    return Ad, Bd, residual
end

# Propagate a control profile through the nonlinear dynamics. Used to build a
# target that is reachable by construction: the generating profile is itself a
# feasible witness, so a run that fails to converge is a solver or algorithm
# result and not an artefact of an impossible instance.
function propagate(
    initial_state::Vector{Float64},
    controls::Matrix{Float64},
    step::Float64,
)
    horizon = size(controls, 2)
    trajectory = zeros(length(initial_state), horizon + 1)
    trajectory[:, 1] = initial_state
    for k in 1:horizon
        trajectory[:, k + 1] = rk4_step(trajectory[:, k], controls[:, k], step)
    end
    return trajectory
end

# A gentle tangential push, which raises the orbit the way a low-thrust
# transfer does.
function tangential_profile(
    initial_state::Vector{Float64},
    horizon::Int,
    step::Float64,
    magnitude::Float64,
)
    controls = zeros(CONTROL_DIMENSION, horizon)
    state = copy(initial_state)
    for k in 1:horizon
        v = state[3:4]
        speed = norm(v)
        controls[:, k] = speed > 0 ? magnitude .* v ./ speed : [magnitude, 0.0]
        state = rk4_step(state, controls[:, k], step)
    end
    return controls
end

struct SCPResult
    solver::String
    converged::Bool
    outer_iterations::Int
    accepted_steps::Int
    rejected_steps::Int
    subproblem_solves::Int
    total_time_sec::Float64
    subproblem_time_sec::Float64
    model_update_time_sec::Float64
    subproblem_times::Vector{Float64}
    subproblem_statuses::Vector{String}
    control_effort::Float64
    max_dynamics_defect::Float64
    endpoint_error::Float64
    max_thrust_violation::Float64
    trajectory::Matrix{Float64}
    controls::Matrix{Float64}
end

# Independent verification: fly the returned controls through the nonlinear
# dynamics with a finer integrator than the one used in the linearization and
# measure how far the result drifts from the reported trajectory.
function verify(trajectory::Matrix{Float64}, controls::Matrix{Float64}, step::Float64, target::Vector{Float64})
    horizon = size(controls, 2)
    state = trajectory[:, 1]
    worst = 0.0
    for k in 1:horizon
        substeps = 8
        for _ in 1:substeps
            state = rk4_step(state, controls[:, k], step / substeps)
        end
        worst = max(worst, norm(state .- trajectory[:, k + 1], Inf))
    end
    return worst, norm(state .- target, Inf)
end

"""
    run_scp(optimizer_factory; ...)

Run the outer sequential convex programming loop with the given optimizer.
The outer algorithm is identical for every solver: the only thing that
changes is which solver answers the convex subproblems.
"""
function run_scp(
    optimizer_factory;
    solver_name::AbstractString = "solver",
    horizon::Int = 40,
    step::Float64 = 0.15,
    thrust_limit::Float64 = 0.05,
    initial_state::Vector{Float64} = [1.0, 0.0, 0.0, 1.0],
    target_thrust::Float64 = 0.03,
    max_outer::Int = 25,
    initial_radius::Float64 = 0.5,
    minimum_radius::Float64 = 1e-5,
    penalty_weight::Float64 = 1e3,
    # Above the RK4 discretization floor of the chosen step, so that the loop
    # stops when the nonlinear problem is solved rather than grinding the
    # trust region down against integration error.
    defect_tolerance::Float64 = 1e-3,
    direct::Bool = true,
    verbose::Bool = false,
)
    nx = STATE_DIMENSION
    nu = CONTROL_DIMENSION

    # The target is where a gentle tangential burn actually gets to, so the
    # instance is feasible by construction and the generating profile stays
    # inside the thrust limit.
    witness_controls = tangential_profile(initial_state, horizon, step, target_thrust)
    target_state = propagate(initial_state, witness_controls, step)[:, end]

    # Start from a dynamically consistent coast: the uncontrolled trajectory.
    # A reference that already satisfies the dynamics is the usual starting
    # point and keeps the first linearization meaningful.
    reference_controls = zeros(nu, horizon)
    reference = propagate(initial_state, reference_controls, step)

    optimizer = optimizer_factory()
    model = direct ? JuMP.direct_model(optimizer) : JuMP.Model(() -> optimizer)
    set_silent(model)

    @variable(model, x[1:nx, 1:(horizon + 1)])
    @variable(model, u[1:nu, 1:horizon])
    # Virtual controls, split into nonnegative parts so the penalty stays
    # linear and the subproblem stays a second-order cone program.
    @variable(model, vplus[1:nx, 1:horizon] >= 0)
    @variable(model, vminus[1:nx, 1:horizon] >= 0)
    # The endpoint condition is penalized rather than imposed. A hard terminal
    # equality together with a trust region makes the very first subproblem
    # infeasible whenever the reference does not already reach the target,
    # which is exactly the situation an outer loop starts in.
    @variable(model, tplus[1:nx] >= 0)
    @variable(model, tminus[1:nx] >= 0)
    @variable(model, radius_slack[1:(horizon + 1)] >= 0)

    initial = @constraint(model, [i = 1:nx], x[i, 1] == initial_state[i])
    terminal = @constraint(
        model,
        [i = 1:nx],
        x[i, horizon + 1] - tplus[i] + tminus[i] == target_state[i],
    )
    dynamics_constraints = Matrix{ConstraintRef}(undef, nx, horizon)
    for k in 1:horizon, i in 1:nx
        dynamics_constraints[i, k] = @constraint(
            model,
            x[i, k + 1] - sum(0.0 * x[j, k] for j in 1:nx) -
            sum(0.0 * u[j, k] for j in 1:nu) - vplus[i, k] + vminus[i, k] == 0.0
        )
    end
    thrust = [
        @constraint(model, [thrust_limit; u[1, k]; u[2, k]] in SecondOrderCone())
        for k in 1:horizon
    ]
    trust = [
        @constraint(
            model,
            vcat(radius_slack[k], [x[i, k] - reference[i, k] for i in 1:nx]) in SecondOrderCone(),
        )
        for k in 1:(horizon + 1)
    ]
    radius_bound = @constraint(model, [k = 1:(horizon + 1)], radius_slack[k] <= initial_radius)

    penalty = penalty_weight
    @objective(
        model,
        Min,
        step * sum(u[j, k]^2 for j in 1:nu, k in 1:horizon) +
        penalty * sum(vplus[i, k] + vminus[i, k] for i in 1:nx, k in 1:horizon) +
        penalty * sum(tplus[i] + tminus[i] for i in 1:nx)
    )

    radius = initial_radius
    accepted = 0
    rejected = 0
    solves = 0
    subproblem_times = Float64[]
    statuses = String[]
    update_time = 0.0
    converged = false
    best_trajectory = copy(reference)
    best_controls = copy(reference_controls)
    # The merit is measured on the nonlinear problem: how far the trajectory
    # drifts when actually flown, plus how far it lands from the target. The
    # coasting reference already scores well on the first term and badly on
    # the second, so a candidate has to improve the real problem to be
    # accepted, not merely the linearized one.
    initial_defect, initial_endpoint = verify(reference, reference_controls, step, target_state)
    best_merit = initial_defect + initial_endpoint

    total_start = time_ns()
    for outer in 1:max_outer
        update_start = time_ns()
        # Relinearize about the current reference. Every dynamics coefficient
        # and right-hand side changes.
        for k in 1:horizon
            Ad, Bd, residual = discrete_linearization(reference[:, k], reference_controls[:, k], step)
            for i in 1:nx
                constraint = dynamics_constraints[i, k]
                for j in 1:nx
                    set_normalized_coefficient(constraint, x[j, k], -Ad[i, j])
                end
                for j in 1:nu
                    set_normalized_coefficient(constraint, u[j, k], -Bd[i, j])
                end
                set_normalized_rhs(constraint, residual[i])
            end
        end
        # The trust-region cone keeps its coefficients; only its centre (the
        # reference trajectory) and its radius move.
        for k in 1:(horizon + 1)
            set_normalized_rhs(radius_bound[k], radius)
            MOI.modify(
                backend(model),
                JuMP.index(trust[k]),
                MOI.VectorConstantChange(
                    vcat(0.0, [-reference[i, k] for i in 1:nx]),
                ),
            )
        end
        for variable in Iterators.flatten((vplus, vminus, tplus, tminus))
            set_objective_coefficient(model, variable, penalty)
        end
        update_time += (time_ns() - update_start) * 1e-9

        solve_start = time_ns()
        optimize!(model)
        elapsed = (time_ns() - solve_start) * 1e-9
        solves += 1
        push!(subproblem_times, elapsed)
        status = termination_status(model)
        push!(statuses, string(status))
        if !(status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)) || result_count(model) < 1
            # A failed subproblem is a rejected step: shrink and retry.
            rejected += 1
            radius *= 0.5
            radius <= minimum_radius && break
            continue
        end

        candidate = value.(x)
        candidate_controls = value.(u)
        defect, endpoint = verify(candidate, candidate_controls, step, target_state)
        merit = defect + endpoint
        verbose && @printf(
            "  outer %2d radius=%.4f merit=%.3e defect=%.3e endpoint=%.3e time=%.4f s status=%s\n",
            outer, radius, merit, defect, endpoint, elapsed, status,
        )
        if merit < best_merit
            accepted += 1
            best_merit = merit
            best_trajectory = candidate
            best_controls = candidate_controls
            reference = candidate
            reference_controls = candidate_controls
            radius = min(radius * 1.6, 4 * initial_radius)
            penalty = min(penalty * 1.5, 1e7)
            if merit < defect_tolerance
                converged = true
                break
            end
        else
            rejected += 1
            radius *= 0.5
            penalty = min(penalty * 2.0, 1e7)
            radius <= minimum_radius && break
        end
    end
    total_time = (time_ns() - total_start) * 1e-9

    defect, endpoint = verify(best_trajectory, best_controls, step, target_state)
    thrust_violation = maximum(
        norm(best_controls[:, k]) - thrust_limit for k in 1:horizon;
        init = -thrust_limit,
    )
    effort = step * sum(abs2, best_controls)

    return SCPResult(
        String(solver_name), converged, accepted + rejected, accepted, rejected, solves,
        total_time, sum(subproblem_times), update_time, subproblem_times, statuses,
        effort, defect, endpoint, max(thrust_violation, 0.0),
        best_trajectory, best_controls,
    )
end

scp_environment() = Dict(
    "julia_version" => string(VERSION),
    "jump_version" => string(pkgversion(JuMP)),
)

end # module NonlinearSCP
