# Solver adapters for the replay benchmark.
#
# Every adapter is driven through the same three operations - `setup!`,
# `update!`, `solve!` - and reports its answer in one common convention, the
# same one the oracle checks:
#
#     minimize (1/2) x' P x + c' x   s.t.  A x = b,   h - G x in K
#
# with dual y on the equalities and z on the cone rows. Where a solver uses a
# different convention, the translation is written out here rather than being
# hidden inside a timing loop.

module Adapters

using LinearAlgebra
using SparseArrays

import MathOptInterface as MOI
import JuliaQOCO

using ..Fixtures

# Reference solvers are optional benchmark dependencies. Binding them at load
# time keeps their constructors callable from code compiled later in the same
# session, which a runtime `Base.require` would not.
const HAS_CLARABEL = try
    @eval import Clarabel
    true
catch
    false
end

const HAS_C_QOCO = try
    @eval import QOCO
    true
catch
    false
end

export SolverAdapter, NativeAdapter, MoiAdapter, ClarabelAdapter
export adapter_name, supports_fixed_pattern_update, setup!, update!, solve!
export solution_of, stats_of, clarabel_available

abstract type SolverAdapter end

adapter_name(adapter::SolverAdapter) = adapter.name
supports_fixed_pattern_update(::SolverAdapter) = true

# --------------------------------------------------------------------------
# JuliaQOCO, native core interface.
# --------------------------------------------------------------------------

mutable struct NativeAdapter <: SolverAdapter
    name::String
    settings::JuliaQOCO.Settings{Float64}
    solver::Any
end

function NativeAdapter(name::AbstractString; kwargs...)
    settings = JuliaQOCO.Settings{Float64}(; verbose = false, kwargs...)
    return NativeAdapter(String(name), settings, nothing)
end

function setup!(adapter::NativeAdapter, instance::ConicInstance)
    adapter.solver = JuliaQOCO.CoreSolver(
        instance.P, instance.c, instance.A, instance.b,
        instance.G, instance.h, instance.l, instance.q;
        settings = JuliaQOCO.copy_settings(adapter.settings),
    )
    return nothing
end

function update!(
    adapter::NativeAdapter,
    instance::ConicInstance;
    matrices::Bool = true,
    objective::Bool = true,
    vectors::Bool = true,
)
    JuliaQOCO.update_data!(
        adapter.solver;
        Px = matrices && objective ? instance.P.nzval : nothing,
        Ax = matrices ? instance.A.nzval : nothing,
        Gx = matrices ? instance.G.nzval : nothing,
        c = objective ? instance.c : nothing,
        b = vectors ? instance.b : nothing,
        h = vectors ? instance.h : nothing,
    )
    return nothing
end

solve!(adapter::NativeAdapter) = (JuliaQOCO._solve!(adapter.solver); nothing)

function solution_of(adapter::NativeAdapter)
    solution = adapter.solver.solution
    return (x = solution.x, s = solution.s, y = solution.y, z = solution.z)
end

function stats_of(adapter::NativeAdapter)
    solution = adapter.solver.solution
    factor = adapter.solver.linsys.factor
    return (
        status = string(solution.status),
        iterations = solution.iters,
        result_iteration = solution.result_iter,
        result_available = solution.result_available,
        objective = solution.obj,
        solve_time_sec = solution.solve_time_sec,
        setup_time_sec = solution.setup_time_sec,
        factorizations = solution.profile.nt_refactors,
        refinements = solution.profile.linsys_refinements,
        regularized_pivots = solution.profile.regularized_pivots,
        factorization_retries = solution.profile.factorization_retries,
        warmstart_accepted = solution.profile.warmstart_accepted,
        warmstart_rejected = solution.profile.warmstart_rejected,
        warmstart_retries = solution.profile.warmstart_retries,
        kkt_nnz = factor === nothing ? 0 : nnz(factor.workspace.triuA),
        factor_nnz = factor === nothing ? 0 : nnz(factor.L),
    )
end

# --------------------------------------------------------------------------
# Any MathOptInterface optimizer, driven through the supported public route.
# --------------------------------------------------------------------------

mutable struct MoiAdapter <: SolverAdapter
    name::String
    factory::Any
    optimizer::Any
    variables::Vector{MOI.VariableIndex}
    equality::Union{Nothing,MOI.ConstraintIndex}
    orthant::Union{Nothing,MOI.ConstraintIndex}
    cones::Vector{MOI.ConstraintIndex}
    cone_ranges::Vector{UnitRange{Int}}
    rebuild_each_solve::Bool
    instance::Union{Nothing,ConicInstance{Float64}}
end

function MoiAdapter(name::AbstractString, factory; rebuild_each_solve::Bool = false)
    return MoiAdapter(
        String(name), factory, nothing, MOI.VariableIndex[], nothing, nothing,
        MOI.ConstraintIndex[], UnitRange{Int}[], rebuild_each_solve, nothing,
    )
end

supports_fixed_pattern_update(adapter::MoiAdapter) = !adapter.rebuild_each_solve

function _vector_affine(
    M::SparseMatrixCSC{Float64,Int},
    rows::UnitRange{Int},
    variables::Vector{MOI.VariableIndex},
    scale::Float64,
    constants::Vector{Float64},
)
    terms = MOI.VectorAffineTerm{Float64}[]
    for column in 1:size(M, 2)
        for position in M.colptr[column]:(M.colptr[column + 1] - 1)
            row = M.rowval[position]
            row in rows || continue
            push!(terms, MOI.VectorAffineTerm(
                row - first(rows) + 1,
                MOI.ScalarAffineTerm(scale * M.nzval[position], variables[column]),
            ))
        end
    end
    return MOI.VectorAffineFunction(terms, constants)
end

function setup!(adapter::MoiAdapter, instance::ConicInstance)
    optimizer = adapter.factory()
    n = length(instance.c)
    variables = MOI.add_variables(optimizer, n)
    adapter.optimizer = optimizer
    adapter.variables = variables
    adapter.instance = instance

    # A x - b in Zeros.
    adapter.equality = MOI.add_constraint(
        optimizer,
        _vector_affine(instance.A, 1:length(instance.b), variables, 1.0, -copy(instance.b)),
        MOI.Zeros(length(instance.b)),
    )
    # h - G x in Nonnegatives, then one SOC per block.
    adapter.cones = MOI.ConstraintIndex[]
    adapter.cone_ranges = UnitRange{Int}[]
    if instance.l > 0
        rows = 1:instance.l
        adapter.orthant = MOI.add_constraint(
            optimizer,
            _vector_affine(instance.G, rows, variables, -1.0, instance.h[rows]),
            MOI.Nonnegatives(instance.l),
        )
        push!(adapter.cone_ranges, rows)
    else
        adapter.orthant = nothing
    end
    offset = instance.l
    for dimension in instance.q
        rows = (offset + 1):(offset + dimension)
        push!(
            adapter.cones,
            MOI.add_constraint(
                optimizer,
                _vector_affine(instance.G, rows, variables, -1.0, instance.h[rows]),
                MOI.SecondOrderCone(dimension),
            ),
        )
        push!(adapter.cone_ranges, rows)
        offset += dimension
    end

    _set_moi_objective!(adapter, instance)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return nothing
end

function _set_moi_objective!(adapter::MoiAdapter, instance::ConicInstance)
    quadratic = MOI.ScalarQuadraticTerm{Float64}[]
    for column in 1:size(instance.P, 2)
        for position in instance.P.colptr[column]:(instance.P.colptr[column + 1] - 1)
            row = instance.P.rowval[position]
            push!(quadratic, MOI.ScalarQuadraticTerm(
                instance.P.nzval[position],
                adapter.variables[row],
                adapter.variables[column],
            ))
        end
    end
    affine = [
        MOI.ScalarAffineTerm(instance.c[i], adapter.variables[i])
        for i in eachindex(instance.c) if instance.c[i] != 0
    ]
    MOI.set(
        adapter.optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction(quadratic, affine, 0.0),
    )
    return nothing
end

function update!(
    adapter::MoiAdapter,
    instance::ConicInstance;
    matrices::Bool = true,
    objective::Bool = true,
    vectors::Bool = true,
)
    if adapter.rebuild_each_solve
        setup!(adapter, instance)
        return nothing
    end
    optimizer = adapter.optimizer
    if matrices || vectors
        MOI.set(
            optimizer,
            MOI.ConstraintFunction(),
            adapter.equality,
            _vector_affine(
                instance.A, 1:length(instance.b), adapter.variables, 1.0, -copy(instance.b),
            ),
        )
        index = 1
        if adapter.orthant !== nothing
            rows = adapter.cone_ranges[index]
            MOI.set(
                optimizer,
                MOI.ConstraintFunction(),
                adapter.orthant,
                _vector_affine(instance.G, rows, adapter.variables, -1.0, instance.h[rows]),
            )
            index += 1
        end
        for cone in adapter.cones
            rows = adapter.cone_ranges[index]
            MOI.set(
                optimizer,
                MOI.ConstraintFunction(),
                cone,
                _vector_affine(instance.G, rows, adapter.variables, -1.0, instance.h[rows]),
            )
            index += 1
        end
    end
    objective && _set_moi_objective!(adapter, instance)
    adapter.instance = instance
    return nothing
end

solve!(adapter::MoiAdapter) = (MOI.optimize!(adapter.optimizer); nothing)

# MOI duals use the convention that the Lagrangian subtracts dual' f(x). For
# `A x - b in Zeros` that gives a contribution of -A' y_moi to stationarity,
# and for `h - G x in K` it gives +G' z_moi, so y = -y_moi and z = z_moi map
# onto the convention used here.
function solution_of(adapter::MoiAdapter)
    optimizer = adapter.optimizer
    instance = adapter.instance
    n = length(instance.c)
    if MOI.get(optimizer, MOI.ResultCount()) < 1
        return (x = fill(NaN, n), s = fill(NaN, length(instance.h)),
                y = fill(NaN, length(instance.b)), z = fill(NaN, length(instance.h)))
    end
    x = [MOI.get(optimizer, MOI.VariablePrimal(), v) for v in adapter.variables]
    y = -MOI.get(optimizer, MOI.ConstraintDual(), adapter.equality)
    s = zeros(length(instance.h))
    z = zeros(length(instance.h))
    index = 1
    if adapter.orthant !== nothing
        rows = adapter.cone_ranges[index]
        s[rows] = MOI.get(optimizer, MOI.ConstraintPrimal(), adapter.orthant)
        z[rows] = MOI.get(optimizer, MOI.ConstraintDual(), adapter.orthant)
        index += 1
    end
    for cone in adapter.cones
        rows = adapter.cone_ranges[index]
        s[rows] = MOI.get(optimizer, MOI.ConstraintPrimal(), cone)
        z[rows] = MOI.get(optimizer, MOI.ConstraintDual(), cone)
        index += 1
    end
    return (x = x, s = s, y = y, z = z)
end

function stats_of(adapter::MoiAdapter)
    optimizer = adapter.optimizer
    iterations = try
        MOI.get(optimizer, MOI.BarrierIterations())
    catch
        -1
    end
    return (
        status = string(MOI.get(optimizer, MOI.TerminationStatus())),
        iterations = iterations,
        result_available = MOI.get(optimizer, MOI.ResultCount()) >= 1,
        objective = MOI.get(optimizer, MOI.ResultCount()) >= 1 ?
                    MOI.get(optimizer, MOI.ObjectiveValue()) : NaN,
        solve_time_sec = try
            MOI.get(optimizer, MOI.SolveTimeSec())
        catch
            NaN
        end,
    )
end

# --------------------------------------------------------------------------
# Clarabel, native interface with its documented fixed-pattern update route.
# --------------------------------------------------------------------------

clarabel_available() = HAS_CLARABEL
c_qoco_available() = HAS_C_QOCO

clarabel_optimizer_factory() = () -> Clarabel.Optimizer()

# The C QOCO wrapper does not support incremental model construction, so it has
# to be driven through a caching layer. That caching layer is part of what is
# being measured, which is why this configuration is labelled as a
# wrapper-level, rebuild-each-solve comparison rather than a measurement of the
# C core in isolation.
function c_qoco_optimizer_factory()
    return () -> begin
        optimizer = MOI.instantiate(() -> QOCO.Optimizer(); with_cache_type = Float64)
        MOI.set(optimizer, MOI.Silent(), true)
        optimizer
    end
end

mutable struct ClarabelAdapter <: SolverAdapter
    name::String
    solver::Any
    cones::Any
    reuse::Bool
    instance::Union{Nothing,ConicInstance{Float64}}
    settings_kwargs::Dict{Symbol,Any}
end

function ClarabelAdapter(name::AbstractString; reuse::Bool = true, kwargs...)
    return ClarabelAdapter(
        String(name), nothing, nothing, reuse, nothing, Dict{Symbol,Any}(kwargs),
    )
end

supports_fixed_pattern_update(adapter::ClarabelAdapter) = adapter.reuse

# Clarabel solves  min (1/2) x'Px + q'x  s.t.  A x + s = b,  s in K, so the
# equality and cone blocks are stacked into one matrix with a zero cone in
# front. Its documented data-update route requires presolve and chordal
# decomposition off, and zero entries must not be dropped, or the pattern the
# update writes into is no longer the pattern that was factorized.
function _clarabel_data(instance::ConicInstance)
    A = vcat(instance.A, instance.G)
    b = vcat(instance.b, instance.h)
    cones = Clarabel.SupportedCone[Clarabel.ZeroConeT(length(instance.b))]
    instance.l > 0 && push!(cones, Clarabel.NonnegativeConeT(instance.l))
    for dimension in instance.q
        push!(cones, Clarabel.SecondOrderConeT(dimension))
    end
    return A, b, cones
end

function setup!(adapter::ClarabelAdapter, instance::ConicInstance)
    A, b, cones = _clarabel_data(instance)
    settings = Clarabel.Settings(;
        verbose = false,
        presolve_enable = false,
        chordal_decomposition_enable = false,
        input_sparse_dropzeros = false,
        adapter.settings_kwargs...,
    )
    solver = Clarabel.Solver()
    Clarabel.setup!(solver, instance.P, instance.c, A, b, cones, settings)
    adapter.solver = solver
    adapter.cones = cones
    adapter.instance = instance
    return nothing
end

function update!(
    adapter::ClarabelAdapter,
    instance::ConicInstance;
    matrices::Bool = true,
    objective::Bool = true,
    vectors::Bool = true,
)
    if !adapter.reuse
        setup!(adapter, instance)
        return nothing
    end
    A, b, _ = _clarabel_data(instance)
    Clarabel.update_data!(
        adapter.solver,
        matrices && objective ? instance.P.nzval : nothing,
        objective ? instance.c : nothing,
        matrices ? A.nzval : nothing,
        vectors ? b : nothing,
    )
    adapter.instance = instance
    return nothing
end

function solve!(adapter::ClarabelAdapter)
    Clarabel.solve!(adapter.solver)
    return nothing
end

function solution_of(adapter::ClarabelAdapter)
    result = adapter.solver.solution
    instance = adapter.instance
    p = length(instance.b)
    m = length(instance.h)
    return (
        x = copy(result.x),
        s = result.s[(p + 1):(p + m)],
        y = copy(result.z[1:p]),
        z = result.z[(p + 1):(p + m)],
    )
end

function stats_of(adapter::ClarabelAdapter)
    result = adapter.solver.solution
    return (
        status = string(result.status),
        iterations = result.iterations,
        result_available = true,
        objective = result.obj_val,
        solve_time_sec = result.solve_time,
    )
end

end # module Adapters
