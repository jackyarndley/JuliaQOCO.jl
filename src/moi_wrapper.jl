Base.@kwdef struct ConstraintInfo{T<:AbstractFloat}
    kind::Symbol
    offset::Int
    length::Int
    sign::T = one(T)
end

mutable struct Optimizer{T<:AbstractFloat} <: MOI.AbstractOptimizer
    settings::Settings{T}
    model::MOIU.UniversalFallback{MOIU.Model{T}}
    native_solver::Any
    constraint_info::Dict{Any,ConstraintInfo{T}}
    objective_constant::T
    objective_sign::T
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    raw_status_string::String
    variable_to_column::Dict{MOI.VariableIndex,Int}
    column_to_variable::Vector{MOI.VariableIndex}
end

function Optimizer{T}(; kwargs...) where {T<:AbstractFloat}
    settings = Settings{T}(; kwargs...)
    model = MOIU.UniversalFallback(MOIU.Model{T}())
    return Optimizer{T}(
        settings,
        model,
        nothing,
        Dict{Any,ConstraintInfo{T}}(),
        zero(T),
        one(T),
        MOI.OPTIMIZE_NOT_CALLED,
        MOI.NO_SOLUTION,
        MOI.NO_SOLUTION,
        status_message(QOCO_UNSOLVED),
        Dict{MOI.VariableIndex,Int}(),
        MOI.VariableIndex[],
    )
end

Optimizer(; kwargs...) = Optimizer{Float64}(; kwargs...)

function _clear_results!(opt::Optimizer{T}) where {T<:AbstractFloat}
    empty!(opt.constraint_info)
    opt.objective_constant = zero(T)
    opt.objective_sign = one(T)
    opt.termination_status = MOI.OPTIMIZE_NOT_CALLED
    opt.primal_status = MOI.NO_SOLUTION
    opt.dual_status = MOI.NO_SOLUTION
    opt.raw_status_string = status_message(QOCO_UNSOLVED)
    return nothing
end

function _reset_results!(opt::Optimizer{T}) where {T<:AbstractFloat}
    opt.native_solver = nothing
    empty!(opt.variable_to_column)
    empty!(opt.column_to_variable)
    return _clear_results!(opt)
end

@inline function _structural_edit_message()
    return "Structural edits after optimize should go through a caching layer so the backend can reset and rebuild efficiently on the next solve."
end

@inline function _throw_if_structural_edit_attached(opt::Optimizer)
    opt.native_solver === nothing || throw(MOI.AddVariableNotAllowed(_structural_edit_message()))
    return nothing
end

MOI.empty!(opt::Optimizer) = (MOI.empty!(opt.model); _reset_results!(opt); nothing)
MOI.is_empty(opt::Optimizer) = MOI.is_empty(opt.model)
MOI.supports_incremental_interface(::Optimizer) = true
Base.summary(io::IO, ::Optimizer{T}) where {T} = print(io, "JuliaQOCO.Optimizer{$T}")

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T<:AbstractFloat}
    MOI.empty!(dest)
    return MOI.copy_to(dest.model, src)
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.VariableIndex}) where {T} = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}) where {T} = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}) where {T} = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true
MOI.supports(::Optimizer, ::MOI.SolverVersion) = true
MOI.supports(::Optimizer, ::MOI.BarrierIterations) = true
MOI.supports(::Optimizer{T}, ::MOI.ConstraintSet, ::Type{MOI.ConstraintIndex{MOI.VectorAffineFunction{T},MOI.Zeros}}) where {T} = false
MOI.supports(::Optimizer{T}, ::MOI.ConstraintSet, ::Type{MOI.ConstraintIndex{MOI.VectorAffineFunction{T},MOI.Nonnegatives}}) where {T} = false
MOI.supports(::Optimizer{T}, ::MOI.ConstraintSet, ::Type{MOI.ConstraintIndex{MOI.VectorAffineFunction{T},MOI.SecondOrderCone}}) where {T} = false
MOI.supports(::Optimizer{T}, ::MOI.ConstraintSet, ::Type{MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Zeros}}) where {T} = false
MOI.supports(::Optimizer{T}, ::MOI.ConstraintSet, ::Type{MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Nonnegatives}}) where {T} = false
MOI.supports(::Optimizer{T}, ::MOI.ConstraintSet, ::Type{MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.SecondOrderCone}}) where {T} = false

MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorAffineFunction{T}}, ::Type{MOI.Zeros}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorAffineFunction{T}}, ::Type{MOI.Nonnegatives}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorAffineFunction{T}}, ::Type{MOI.SecondOrderCone}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorOfVariables}, ::Type{MOI.Zeros}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorOfVariables}, ::Type{MOI.Nonnegatives}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorOfVariables}, ::Type{MOI.SecondOrderCone}) where {T} = true

MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool) = (opt.settings.verbose = !value)
MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.settings.verbose

function MOI.set(opt::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    name = Symbol(attr.name)
    if name in fieldnames(typeof(opt.settings))
        converted = convert(fieldtype(typeof(opt.settings), name), value)
        setfield!(opt.settings, name, converted)
        return
    end
    throw(ArgumentError("Unknown JuliaQOCO setting $(attr.name)"))
end

function MOI.get(opt::Optimizer, attr::MOI.RawOptimizerAttribute)
    name = Symbol(attr.name)
    if name in fieldnames(typeof(opt.settings))
        return getfield(opt.settings, name)
    end
    throw(ArgumentError("Unknown JuliaQOCO setting $(attr.name)"))
end

MOI.get(::Optimizer, ::MOI.SolverName) = "JuliaQOCO"
MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.1.0"
MOI.get(opt::Optimizer, ::MOI.BarrierIterations) = opt.native_solver === nothing ? 0 : opt.native_solver.solution.iters
MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt.native_solver
MOI.get(opt::Optimizer, ::MOI.RawStatusString) = opt.raw_status_string
MOI.get(opt::Optimizer, ::MOI.TerminationStatus) = opt.termination_status
MOI.get(opt::Optimizer, ::MOI.PrimalStatus) = opt.primal_status
MOI.get(opt::Optimizer, ::MOI.DualStatus) = opt.dual_status
MOI.get(opt::Optimizer, ::MOI.ResultCount) = opt.primal_status == MOI.NO_SOLUTION ? 0 : 1
MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) = opt.native_solver === nothing ? 0.0 : opt.native_solver.solution.solve_time_sec

MOI.is_valid(opt::Optimizer, vi::MOI.VariableIndex) = MOI.is_valid(opt.model, vi)
MOI.is_valid(opt::Optimizer, ci::MOI.ConstraintIndex) = MOI.is_valid(opt.model, ci)

function MOI.add_variable(opt::Optimizer)
    _throw_if_structural_edit_attached(opt)
    vi = MOI.add_variable(opt.model)
    _reset_results!(opt)
    return vi
end

function MOI.add_variables(opt::Optimizer, n::Int)
    _throw_if_structural_edit_attached(opt)
    vis = MOI.add_variables(opt.model, n)
    _reset_results!(opt)
    return vis
end

function MOI.add_constrained_variable(opt::Optimizer, set::S) where {S<:MOI.AbstractScalarSet}
    _throw_if_structural_edit_attached(opt)
    result = MOI.add_constrained_variable(opt.model, set)
    _reset_results!(opt)
    return result
end

function MOI.add_constrained_variables(opt::Optimizer, set::S) where {S<:MOI.AbstractVectorSet}
    _throw_if_structural_edit_attached(opt)
    result = MOI.add_constrained_variables(opt.model, set)
    _reset_results!(opt)
    return result
end

MOI.supports_add_constrained_variable(::Optimizer, ::Type{<:MOI.AbstractScalarSet}) = false
MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Reals}) = false
MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Zeros}) = true
MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Nonnegatives}) = true
MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.SecondOrderCone}) = true

function MOI.add_constraint(
    opt::Optimizer{T},
    func::F,
    set::S,
) where {T<:AbstractFloat,F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    opt.native_solver === nothing || throw(MOI.AddConstraintNotAllowed{F,S}(_structural_edit_message()))
    ci = MOI.add_constraint(opt.model, func, set)
    _reset_results!(opt)
    return ci
end

function MOI.delete(opt::Optimizer, vi::MOI.VariableIndex)
    opt.native_solver === nothing || throw(MOI.DeleteNotAllowed(vi, _structural_edit_message()))
    MOI.delete(opt.model, vi)
    _reset_results!(opt)
    return nothing
end

function MOI.delete(opt::Optimizer, vis::Vector{MOI.VariableIndex})
    if opt.native_solver !== nothing && !isempty(vis)
        throw(MOI.DeleteNotAllowed(first(vis), _structural_edit_message()))
    end
    MOI.delete(opt.model, vis)
    _reset_results!(opt)
    return nothing
end

function MOI.delete(opt::Optimizer, ci::MOI.ConstraintIndex)
    opt.native_solver === nothing || throw(MOI.DeleteNotAllowed(ci, _structural_edit_message()))
    MOI.delete(opt.model, ci)
    _reset_results!(opt)
    return nothing
end

function MOI.delete(opt::Optimizer, cis::Vector{CI}) where {CI<:MOI.ConstraintIndex}
    if opt.native_solver !== nothing && !isempty(cis)
        throw(MOI.DeleteNotAllowed(first(cis), _structural_edit_message()))
    end
    MOI.delete(opt.model, cis)
    _reset_results!(opt)
    return nothing
end

MOI.get(opt::Optimizer, attr::MOI.AbstractModelAttribute) = MOI.get(opt.model, attr)
MOI.get(opt::Optimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex) = MOI.get(opt.model, attr, vi)
MOI.get(opt::Optimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex) = MOI.get(opt.model, attr, ci)

function MOI.set(opt::Optimizer, attr::MOI.AbstractModelAttribute, value)
    result = MOI.set(opt.model, attr, value)
    _reset_results!(opt)
    return result
end

function MOI.set(opt::Optimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex, value)
    result = MOI.set(opt.model, attr, vi, value)
    _reset_results!(opt)
    return result
end

function MOI.set(opt::Optimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex, value)
    result = MOI.set(opt.model, attr, ci, value)
    _reset_results!(opt)
    return result
end

function _variable_columns(model::MOIU.UniversalFallback{MOIU.Model{T}}) where {T<:AbstractFloat}
    variables = MOI.get(model, MOI.ListOfVariableIndices())
    variable_to_column = Dict{MOI.VariableIndex,Int}()
    sizehint!(variable_to_column, length(variables))
    for (i, vi) in enumerate(variables)
        variable_to_column[vi] = i
    end
    return variables, variable_to_column
end

function _constraint_value(
    x::AbstractVector{T},
    f::MOI.VectorOfVariables,
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
) where {T<:AbstractFloat}
    return T[x[variable_to_column[v]] for v in f.variables]
end

function _constraint_value(
    x::AbstractVector{T},
    f::MOI.VectorAffineFunction{T},
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
) where {T<:AbstractFloat}
    y = copy(f.constants)
    for term in f.terms
        y[term.output_index] += term.scalar_term.coefficient * x[variable_to_column[term.scalar_term.variable]]
    end
    return y
end

function _constraint_value(
    x::AbstractVector{T},
    f::MOI.VariableIndex,
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
) where {T<:AbstractFloat}
    return x[variable_to_column[f]]
end

function _constraint_value(
    x::AbstractVector{T},
    f::MOI.ScalarAffineFunction{T},
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
) where {T<:AbstractFloat}
    y = f.constant
    for term in f.terms
        y += term.coefficient * x[variable_to_column[term.variable]]
    end
    return y
end

function _objective_data(
    model::MOIU.UniversalFallback{MOIU.Model{T}},
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
    n::Int,
) where {T<:AbstractFloat}
    rows = Int[]
    cols = Int[]
    vals = T[]
    c = zeros(T, n)
    constant = zero(T)

    sense = MOI.get(model, MOI.ObjectiveSense())
    sign = sense == MOI.MAX_SENSE ? -one(T) : one(T)

    F = MOI.get(model, MOI.ObjectiveFunctionType())
    if F == MOI.VariableIndex
        vi = MOI.get(model, MOI.ObjectiveFunction{MOI.VariableIndex}())
        c[variable_to_column[vi]] = sign
    elseif F == MOI.ScalarAffineFunction{T}
        f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
        constant = f.constant
        for term in f.terms
            c[variable_to_column[term.variable]] += sign * term.coefficient
        end
    elseif F == MOI.ScalarQuadraticFunction{T}
        f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}())
        constant = f.constant
        sizehint!(rows, length(f.quadratic_terms))
        sizehint!(cols, length(f.quadratic_terms))
        sizehint!(vals, length(f.quadratic_terms))
        for term in f.affine_terms
            c[variable_to_column[term.variable]] += sign * term.coefficient
        end
        for term in f.quadratic_terms
            i = variable_to_column[term.variable_1]
            j = variable_to_column[term.variable_2]
            coeff = sign * term.coefficient
            if i <= j
                push!(rows, i)
                push!(cols, j)
            else
                push!(rows, j)
                push!(cols, i)
            end
            push!(vals, coeff)
        end
    end

    P = sparse(rows, cols, vals, n, n)
    return P, c, constant, sign
end

function _append_scaled!(dest::Vector{T}, src::AbstractVector{T}, α::T) where {T<:AbstractFloat}
    start = length(dest) + 1
    resize!(dest, start + length(src) - 1)
    @inbounds for i in eachindex(src)
        dest[start + i - 1] = α * src[i]
    end
    return dest
end

function _append_zeros!(dest::Vector{T}, n::Integer) where {T<:AbstractFloat}
    start = length(dest) + 1
    resize!(dest, start + n - 1)
    @inbounds for i in start:length(dest)
        dest[i] = zero(T)
    end
    return dest
end

function _append_vector_affine_eq!(
    Arows::Vector{Int},
    Acols::Vector{Int},
    Avals::Vector{T},
    b::Vector{T},
    f::MOI.VectorAffineFunction{T},
    row_offset::Int,
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
) where {T<:AbstractFloat}
    _append_scaled!(b, f.constants, -one(T))
    for term in f.terms
        push!(Arows, row_offset + term.output_index)
        push!(Acols, variable_to_column[term.scalar_term.variable])
        push!(Avals, term.scalar_term.coefficient)
    end
    return nothing
end

function _append_vector_affine_cone!(
    Grows::Vector{Int},
    Gcols::Vector{Int},
    Gvals::Vector{T},
    h::Vector{T},
    f::MOI.VectorAffineFunction{T},
    row_offset::Int,
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
) where {T<:AbstractFloat}
    append!(h, f.constants)
    for term in f.terms
        push!(Grows, row_offset + term.output_index)
        push!(Gcols, variable_to_column[term.scalar_term.variable])
        push!(Gvals, -term.scalar_term.coefficient)
    end
    return nothing
end

function _append_vector_of_variables_eq!(
    Arows::Vector{Int},
    Acols::Vector{Int},
    Avals::Vector{T},
    b::Vector{T},
    f::MOI.VectorOfVariables,
    row_offset::Int,
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
) where {T<:AbstractFloat}
    _append_zeros!(b, length(f.variables))
    for (i, vi) in enumerate(f.variables)
        push!(Arows, row_offset + i)
        push!(Acols, variable_to_column[vi])
        push!(Avals, one(T))
    end
    return nothing
end

function _append_vector_of_variables_cone!(
    Grows::Vector{Int},
    Gcols::Vector{Int},
    Gvals::Vector{T},
    h::Vector{T},
    f::MOI.VectorOfVariables,
    row_offset::Int,
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
) where {T<:AbstractFloat}
    _append_zeros!(h, length(f.variables))
    for (i, vi) in enumerate(f.variables)
        push!(Grows, row_offset + i)
        push!(Gcols, variable_to_column[vi])
        push!(Gvals, -one(T))
    end
    return nothing
end

function _constraint_data(
    opt::Optimizer{T},
    variable_to_column::AbstractDict{MOI.VariableIndex,<:Integer},
    n::Int,
) where {T<:AbstractFloat}
    Arows = Int[]
    Acols = Int[]
    Avals = T[]
    b = T[]
    Grows = Int[]
    Gcols = Int[]
    Gvals = T[]
    h = T[]
    l = 0
    q = Int[]

    vaeq = MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T},MOI.Zeros}())
    vveq = MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MOI.Zeros}())
    vann = MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T},MOI.Nonnegatives}())
    vvnn = MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MOI.Nonnegatives}())
    vasoc = MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T},MOI.SecondOrderCone}())
    vvsoc = MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MOI.SecondOrderCone}())

    info = Dict{Any,ConstraintInfo{T}}()
    sizehint!(info, length(vaeq) + length(vveq) + length(vann) + length(vvnn) + length(vasoc) + length(vvsoc))
    sizehint!(q, length(vasoc) + length(vvsoc))

    eq_offset = 0
    cone_offset = 0

    for ci in vaeq
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.constants)
        _append_vector_affine_eq!(Arows, Acols, Avals, b, f, eq_offset, variable_to_column)
        info[ci] = ConstraintInfo{T}(kind = :eq, offset = eq_offset + 1, length = dim, sign = -one(T))
        eq_offset += dim
    end

    for ci in vveq
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.variables)
        _append_vector_of_variables_eq!(Arows, Acols, Avals, b, f, eq_offset, variable_to_column)
        info[ci] = ConstraintInfo{T}(kind = :eq, offset = eq_offset + 1, length = dim, sign = -one(T))
        eq_offset += dim
    end

    # The native solver expects the cone rows ordered as all nonnegative rows
    # first, followed by SOC blocks in q order.
    for ci in vann
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.constants)
        _append_vector_affine_cone!(Grows, Gcols, Gvals, h, f, cone_offset, variable_to_column)
        info[ci] = ConstraintInfo{T}(kind = :cone, offset = cone_offset + 1, length = dim)
        cone_offset += dim
        l += dim
    end

    for ci in vvnn
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.variables)
        _append_vector_of_variables_cone!(Grows, Gcols, Gvals, h, f, cone_offset, variable_to_column)
        info[ci] = ConstraintInfo{T}(kind = :cone, offset = cone_offset + 1, length = dim)
        cone_offset += dim
        l += dim
    end

    for ci in vasoc
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.constants)
        _append_vector_affine_cone!(Grows, Gcols, Gvals, h, f, cone_offset, variable_to_column)
        info[ci] = ConstraintInfo{T}(kind = :cone, offset = cone_offset + 1, length = dim)
        cone_offset += dim
        push!(q, dim)
    end

    for ci in vvsoc
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.variables)
        _append_vector_of_variables_cone!(Grows, Gcols, Gvals, h, f, cone_offset, variable_to_column)
        info[ci] = ConstraintInfo{T}(kind = :cone, offset = cone_offset + 1, length = dim)
        cone_offset += dim
        push!(q, dim)
    end

    for (F, S) in MOI.get(opt.model, MOI.ListOfConstraintTypesPresent())
        if (F == MOI.VectorAffineFunction{T} && (S == MOI.Zeros || S == MOI.Nonnegatives || S == MOI.SecondOrderCone)) ||
           (F == MOI.VectorOfVariables && (S == MOI.Zeros || S == MOI.Nonnegatives || S == MOI.SecondOrderCone))
            continue
        end
        cis = MOI.get(opt.model, MOI.ListOfConstraintIndices{F,S}())
        if !isempty(cis)
            throw(ArgumentError("Unsupported constraint type $(F) in $(S). Use the bridged optimizer interface."))
        end
    end

    A = sparse(Arows, Acols, Avals, eq_offset, n)
    G = sparse(Grows, Gcols, Gvals, cone_offset, n)
    return A, b, G, h, l, q, info
end

function MOI.optimize!(opt::Optimizer{T}) where {T<:AbstractFloat}
    _reset_results!(opt)
    variables, variable_to_column = _variable_columns(opt.model)
    n = length(variables)
    P, c, constant, sign = _objective_data(opt.model, variable_to_column, n)
    A, b, G, h, l, q, info = _constraint_data(opt, variable_to_column, n)
    solver = Solver(P, c, A, b, G, h, l, q; settings = opt.settings)
    solve!(solver)
    opt.native_solver = solver
    opt.constraint_info = info
    opt.objective_constant = constant
    opt.objective_sign = sign
    opt.raw_status_string = status_string(solver.solution.status, solver.solution.status_detail)
    opt.variable_to_column = variable_to_column
    opt.column_to_variable = variables

    if solver.solution.status == QOCO_SOLVED
        opt.termination_status = MOI.OPTIMAL
        opt.primal_status = MOI.FEASIBLE_POINT
        opt.dual_status = MOI.FEASIBLE_POINT
    elseif solver.solution.status == QOCO_SOLVED_INACCURATE
        opt.termination_status = MOI.ALMOST_OPTIMAL
        opt.primal_status = MOI.FEASIBLE_POINT
        opt.dual_status = MOI.FEASIBLE_POINT
    elseif solver.solution.status == QOCO_MAX_ITER
        opt.termination_status = MOI.ITERATION_LIMIT
        opt.primal_status = MOI.UNKNOWN_RESULT_STATUS
        opt.dual_status = MOI.UNKNOWN_RESULT_STATUS
    else
        opt.termination_status = MOI.NUMERICAL_ERROR
        opt.primal_status = MOI.UNKNOWN_RESULT_STATUS
        opt.dual_status = MOI.UNKNOWN_RESULT_STATUS
    end
    return nothing
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ObjectiveValue) where {T<:AbstractFloat}
    MOI.check_result_index_bounds(opt, attr)
    return opt.objective_sign * opt.native_solver.solution.obj + opt.objective_constant
end

function MOI.get(opt::Optimizer{T}, attr::MOI.VariablePrimal, vi::MOI.VariableIndex) where {T<:AbstractFloat}
    MOI.check_result_index_bounds(opt, attr)
    return opt.native_solver.solution.x[opt.variable_to_column[vi]]
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F,S}) where {T<:AbstractFloat,F,S}
    MOI.check_result_index_bounds(opt, attr)
    info = opt.constraint_info[ci]
    vec = info.kind == :eq ? opt.native_solver.solution.y : opt.native_solver.solution.z
    vals = info.sign .* vec[info.offset:(info.offset + info.length - 1)]
    return F <: MOI.AbstractVectorFunction ? collect(vals) : vals[1]
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F,S}) where {T<:AbstractFloat,F,S}
    MOI.check_result_index_bounds(opt, attr)
    x = opt.native_solver.solution.x
    f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
    return _constraint_value(x, f, opt.variable_to_column)
end
