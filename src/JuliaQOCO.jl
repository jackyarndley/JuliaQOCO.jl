module JuliaQOCO

using LinearAlgebra
using PrecompileTools: @compile_workload
using Printf
using SparseArrays

import MathOptInterface as MOI

include("internal_qdldl.jl")
const QDLDL = InternalQDLDL

const MOIU = MOI.Utilities

export Optimizer

include("statuscodes.jl")
include("settings.jl")
include("types.jl")
include("utils.jl")
include("common_linalg.jl")
include("equilibration.jl")
include("cones.jl")
include("kkt.jl")
include("solver.jl")
include("updates.jl")
include("moi_wrapper.jl")

default_settings(::Type{T} = Float64) where {T<:AbstractFloat} = Settings{T}()

moi_version() = string(pkgversion(MOI))

# The workload deliberately covers the routes a repeated-solve caller actually
# uses - the MathOptInterface layer, a second-order cone, and a fixed-pattern
# matrix update - not just a scalar orthant program, because those are where
# first-use latency was measured to be. The instances stay tiny: the point is
# to compile the code paths, not to specialize on any particular size, and no
# modelling package is pulled in as a runtime dependency to do it.
@compile_workload begin
    settings = Settings{Float64}(; verbose = false)
    P = spzeros(Float64, 1, 1)
    G = sparse([-1.0;;])
    solver = CoreSolver(P, [1.0], nothing, nothing, G, [-1.0], 1, Int[]; settings = settings)
    _solve!(solver)
    update_vector_data!(solver; c = [0.5])
    _solve!(solver)

    # A quadratic objective, an equality, an orthant row and a second-order
    # cone, followed by a fixed-pattern update of every matrix and vector.
    Pq = sparse([1, 2], [1, 2], [2.0, 1.0], 2, 2)
    Aq = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2)
    Gq = sparse([1, 2, 3], [1, 1, 2], [-1.0, -1.0, -1.0], 3, 2)
    conic = CoreSolver(
        Pq, [-1.0, 0.5], Aq, [1.0], Gq, [0.0, 1.0, 0.0], 1, [2];
        settings = settings,
    )
    _solve!(conic)
    update_data!(
        conic;
        Px = Pq.nzval, Ax = Aq.nzval, Gx = Gq.nzval,
        c = [-0.9, 0.4], b = [1.0], h = [0.0, 1.1, 0.0],
    )
    _solve!(conic)

    optimizer = Optimizer{Float64}(; verbose = false)
    variables = MOI.add_variables(optimizer, 2)
    MOI.add_constraint(optimizer, variables[1], MOI.GreaterThan(0.0))
    equality = MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, variables[1]),
             MOI.ScalarAffineTerm(1.0, variables[2])],
            0.0,
        ),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraint(
        optimizer,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, variables[2]))],
            [1.0, 0.0],
        ),
        MOI.SecondOrderCone(2),
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction(
            [MOI.ScalarQuadraticTerm(2.0, variables[1], variables[1])],
            [MOI.ScalarAffineTerm(-1.0, variables[2])],
            0.0,
        ),
    )
    MOI.optimize!(optimizer)
    MOI.get(optimizer, MOI.TerminationStatus())
    MOI.get(optimizer, MOI.VariablePrimal(), variables[1])
    MOI.get(optimizer, MOI.ConstraintDual(), equality)
    MOI.modify(optimizer, equality, MOI.ScalarCoefficientChange(variables[1], 1.5))
    MOI.optimize!(optimizer)
    MOI.get(optimizer, MOI.ObjectiveValue())
end

end
