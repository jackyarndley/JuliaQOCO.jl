# Deterministic problem fixtures for the replay benchmark.
#
# One fixture is a *fixed sparsity pattern* plus a precomputed sequence of
# numerical instances on that pattern. The sequence is generated once, from a
# fixed seed, and then replayed identically by every solver and configuration:
# no solver ever sees a different problem from another, and no benchmarking
# harness can evolve the data between samples.
#
# The shape is a discrete-time optimal-control subproblem of the kind a
# sequential convex programming loop produces:
#
#   variables   z = [x_0; u_0; x_1; u_1; ...; x_{N-1}; u_{N-1}; x_N]
#   equalities  x_0 = x_init,  x_{k+1} = A_k x_k + B_k u_k + w_k
#   cone rows   control boxes (orthant)
#               one thrust cone per stage      (many small SOCs)
#               one global trust region on x   (a single large SOC)
#   objective   diagonal state and control weights
#
# Both second-order-cone regimes matter: the per-stage cones are the case that
# a compact dense Nesterov-Todd block handles well, and the global trust region
# is the case where that block is quadratic in the cone dimension.

module Fixtures

using LinearAlgebra
using Random
using SparseArrays

export ConicInstance, Fixture, build_fixture, instance_at, pattern_summary

# One numerical instance on a fixed pattern. The matrices are stored in full so
# that the oracle can use them directly; the nonzero value vectors are what the
# fixed-pattern update routes consume.
struct ConicInstance{T<:AbstractFloat}
    P::SparseMatrixCSC{T,Int}
    c::Vector{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    G::SparseMatrixCSC{T,Int}
    h::Vector{T}
    l::Int
    q::Vector{Int}
end

struct Fixture{T<:AbstractFloat}
    name::String
    horizon::Int
    nx::Int
    nu::Int
    base::ConicInstance{T}
    sequence::Vector{ConicInstance{T}}
    # Index of the variables of each stage, used by the MOI-level fixtures.
    state_columns::Vector{UnitRange{Int}}
    control_columns::Vector{UnitRange{Int}}
end

_state_offset(k::Int, nx::Int, nu::Int) = k * (nx + nu)
_control_offset(k::Int, nx::Int, nu::Int) = k * (nx + nu) + nx

# Structural pattern plus the numerical values for one linearization. Passing
# the same `rng` state through every instance would make the instances differ
# structurally, so the pattern is built once and only the values change.
function _instance(
    horizon::Int,
    nx::Int,
    nu::Int,
    dynamics::Vector{Matrix{Float64}},
    controls::Vector{Matrix{Float64}},
    drift::Vector{Vector{Float64}},
    x_init::Vector{Float64},
    # A whole reference trajectory, not a single point. Centring the global
    # trust region on a dynamically consistent trajectory is what a sequential
    # convex programming loop actually does, and it keeps the region
    # satisfiable at any horizon: a fixed centre would need a radius growing
    # like the square root of the trajectory length just to stay feasible.
    x_reference::Vector{Vector{Float64}},
    trust_radius::Float64,
    thrust_limit::Float64,
    control_bound::Float64,
    state_weight::Float64,
    control_weight::Float64,
)
    n = (horizon + 1) * nx + horizon * nu

    # --- objective -------------------------------------------------------
    diagonal = zeros(n)
    linear = zeros(n)
    for k in 0:horizon
        offset = _state_offset(k, nx, nu)
        for i in 1:nx
            diagonal[offset + i] = state_weight * (1 + 0.1 * i)
            linear[offset + i] = -0.01 * x_reference[k + 1][i]
        end
    end
    for k in 0:(horizon - 1)
        offset = _control_offset(k, nx, nu)
        for j in 1:nu
            diagonal[offset + j] = control_weight * (1 + 0.05 * j)
        end
    end
    P = sparse(1:n, 1:n, diagonal, n, n)

    # --- equalities ------------------------------------------------------
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    b = zeros((horizon + 1) * nx)
    for i in 1:nx
        push!(rows, i)
        push!(cols, _state_offset(0, nx, nu) + i)
        push!(vals, 1.0)
        b[i] = x_init[i]
    end
    row = nx
    for k in 0:(horizon - 1)
        Ak = dynamics[k + 1]
        Bk = controls[k + 1]
        for i in 1:nx
            row += 1
            push!(rows, row)
            push!(cols, _state_offset(k + 1, nx, nu) + i)
            push!(vals, 1.0)
            for j in 1:nx
                push!(rows, row)
                push!(cols, _state_offset(k, nx, nu) + j)
                push!(vals, -Ak[i, j])
            end
            for j in 1:nu
                push!(rows, row)
                push!(cols, _control_offset(k, nx, nu) + j)
                push!(vals, -Bk[i, j])
            end
            b[row] = drift[k + 1][i]
        end
    end
    A = sparse(rows, cols, vals, (horizon + 1) * nx, n)

    # --- cone rows -------------------------------------------------------
    grows = Int[]
    gcols = Int[]
    gvals = Float64[]
    h = Float64[]
    cone_row = 0
    # Orthant: -bound <= u <= bound.
    for k in 0:(horizon - 1), j in 1:nu
        column = _control_offset(k, nx, nu) + j
        cone_row += 1
        push!(grows, cone_row); push!(gcols, column); push!(gvals, 1.0)
        push!(h, control_bound)
        cone_row += 1
        push!(grows, cone_row); push!(gcols, column); push!(gvals, -1.0)
        push!(h, control_bound)
    end
    l = cone_row
    q = Int[]
    # Many small cones: one thrust limit per stage.
    for k in 0:(horizon - 1)
        cone_row += 1
        push!(h, thrust_limit)
        for j in 1:nu
            cone_row += 1
            push!(grows, cone_row)
            push!(gcols, _control_offset(k, nx, nu) + j)
            push!(gvals, -1.0)
            push!(h, 0.0)
        end
        push!(q, nu + 1)
    end
    # One large cone: a global Euclidean trust region on the whole state
    # trajectory, which is the block whose dense Nesterov-Todd storage grows
    # quadratically.
    cone_row += 1
    push!(h, trust_radius)
    for k in 0:horizon, i in 1:nx
        cone_row += 1
        push!(grows, cone_row)
        push!(gcols, _state_offset(k, nx, nu) + i)
        push!(gvals, -1.0)
        push!(h, -x_reference[k + 1][i])
    end
    push!(q, (horizon + 1) * nx + 1)
    G = sparse(grows, gcols, gvals, cone_row, n)

    return ConicInstance{Float64}(P, linear, A, b, G, h, l, q)
end

"""
    build_fixture(; name, horizon, nx, nu, steps, seed)

Precompute a fixed pattern and a sequence of `steps` numerical instances on
it. Successive instances differ the way a sequential convex programming loop
makes them differ: relinearized dynamics, a moved reference trajectory, a
changed trust radius and changed penalty weights.
"""
function build_fixture(;
    name::AbstractString = "control",
    horizon::Int = 20,
    nx::Int = 6,
    nu::Int = 3,
    steps::Int = 8,
    seed::Int = 20260907,
)
    rng = MersenneTwister(seed)
    stable = Matrix{Float64}(I, nx, nx) + 0.05 * randn(rng, nx, nx)
    input = 0.2 * randn(rng, nx, nu)
    x_init = 0.5 * randn(rng, nx)

    instances = ConicInstance{Float64}[]
    for step in 0:steps
        phase = 0.13 * step
        dynamics = [stable .* (1 + 0.02 * sin(phase + 0.1 * k)) for k in 0:(horizon - 1)]
        controls = [input .* (1 + 0.03 * cos(phase + 0.07 * k)) for k in 0:(horizon - 1)]
        drift = [0.01 * sin.(phase .+ (1:nx) .+ k) for k in 0:(horizon - 1)]
        # The reference is the uncontrolled trajectory of this linearization,
        # so the trust region is centred on a dynamically consistent path and a
        # modest radius stays feasible at every horizon.
        reference = Vector{Vector{Float64}}(undef, horizon + 1)
        reference[1] = copy(x_init)
        for k in 1:horizon
            reference[k + 1] = dynamics[k] * reference[k] .+ drift[k]
        end
        push!(
            instances,
            _instance(
                horizon, nx, nu, dynamics, controls, drift,
                x_init,
                reference,
                2.0 + 0.5 * sin(phase),      # trust radius
                0.8 + 0.05 * cos(phase),     # thrust limit
                1.0,                         # control box
                0.5 + 0.05 * step,           # state weight
                0.1 + 0.01 * step,           # control weight
            ),
        )
    end

    state_columns = [
        (_state_offset(k, nx, nu) + 1):(_state_offset(k, nx, nu) + nx)
        for k in 0:horizon
    ]
    control_columns = [
        (_control_offset(k, nx, nu) + 1):(_control_offset(k, nx, nu) + nu)
        for k in 0:(horizon - 1)
    ]
    return Fixture{Float64}(
        String(name), horizon, nx, nu,
        instances[1], instances[2:end],
        state_columns, control_columns,
    )
end

instance_at(fixture::Fixture, step::Int) =
    step == 0 ? fixture.base : fixture.sequence[step]

function pattern_summary(fixture::Fixture)
    base = fixture.base
    return (
        name = fixture.name,
        horizon = fixture.horizon,
        nx = fixture.nx,
        nu = fixture.nu,
        variables = length(base.c),
        equality_rows = length(base.b),
        cone_rows = length(base.h),
        orthant_rows = base.l,
        soc_count = length(base.q),
        soc_max = isempty(base.q) ? 0 : maximum(base.q),
        nnz_P = nnz(base.P),
        nnz_A = nnz(base.A),
        nnz_G = nnz(base.G),
        steps = length(fixture.sequence),
    )
end

end # module Fixtures
