const _MATRIX_P = UInt8(1)
const _MATRIX_A = UInt8(2)
const _MATRIX_G = UInt8(3)
const _ROW_EQUALITY = UInt8(1)
const _ROW_CONE = UInt8(2)

struct ConstraintKey
    function_kind::UInt8
    set_kind::UInt8
    value::Int
end

struct CoefficientKey
    constraint::ConstraintKey
    output_index::Int
    variable::Int
end

struct QuadraticKey
    row::Int
    column::Int
end

struct ConstraintInfo{T<:AbstractFloat}
    row_kind::UInt8
    offset::Int
    length::Int
    dual_sign::T
end

struct PendingTarget{T<:AbstractFloat}
    matrix::UInt8
    row_1::Int
    column_1::Int
    multiplier_1::T
    row_2::Int
    column_2::Int
    multiplier_2::T
end

struct MatrixTarget{T<:AbstractFloat}
    matrix::UInt8
    position_1::Int
    raw_position_1::Int
    multiplier_1::T
    position_2::Int
    raw_position_2::Int
    multiplier_2::T
end

mutable struct DirtyQueue{T<:AbstractFloat}
    generation::UInt32
    marks::Vector{UInt32}
    indices::Vector{Int}
    values::Vector{T}
    commit_values::Vector{T}
end

function DirtyQueue(values::Vector{T}) where {T<:AbstractFloat}
    return DirtyQueue{T}(UInt32(1), zeros(UInt32, length(values)), Int[], values, T[])
end

@inline function _queue!(queue::DirtyQueue{T}, index::Int, value::T) where {T}
    queue.values[index] = value
    if queue.marks[index] != queue.generation
        queue.marks[index] = queue.generation
        push!(queue.indices, index)
    end
    return nothing
end

function _commit_values!(queue::DirtyQueue)
    empty!(queue.commit_values)
    sizehint!(queue.commit_values, length(queue.indices))
    @inbounds for index in queue.indices
        push!(queue.commit_values, queue.values[index])
    end
    return queue.commit_values
end

function _clear_queue!(queue::DirtyQueue)
    empty!(queue.indices)
    empty!(queue.commit_values)
    if queue.generation == typemax(UInt32)
        fill!(queue.marks, UInt32(0))
        queue.generation = UInt32(1)
    else
        queue.generation += UInt32(1)
    end
    return nothing
end

mutable struct MOIAssemblyCache{T<:AbstractFloat}
    solver::Solver{T,Int,F} where {F}
    constraint_info::Dict{ConstraintKey,ConstraintInfo{T}}
    coefficient_targets::Dict{CoefficientKey,MatrixTarget{T}}
    quadratic_targets::Dict{QuadraticKey,MatrixTarget{T}}
    raw_P::Vector{T}
    raw_A::Vector{T}
    raw_G::Vector{T}
    P_dirty::DirtyQueue{T}
    A_dirty::DirtyQueue{T}
    G_dirty::DirtyQueue{T}
    c_dirty::DirtyQueue{T}
    b_dirty::DirtyQueue{T}
    h_dirty::DirtyQueue{T}
    variable_to_column::Vector{Int}
    column_to_variable::Vector{MOI.VariableIndex}
    start_buffer::Vector{T}
    objective_constant::T
    objective_sign::T
    structure_generation::UInt64
    rebuild_count::Int
    commit_count::Int
    last_commit_time_sec::Float64
end

mutable struct Optimizer{T<:AbstractFloat} <: MOI.AbstractOptimizer
    settings::Settings{T}
    model::MOIU.UniversalFallback{MOIU.Model{T}}
    cache::Union{Nothing,MOIAssemblyCache{T}}
    structure_dirty::Bool
    settings_dirty::Bool
    starts_dirty::Bool
    structure_generation::UInt64
    rebuild_count::Int
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    raw_status_string::String
end

function Optimizer{T}(; kwargs...) where {T<:AbstractFloat}
    settings = Settings{T}(; kwargs...)
    validate_settings(settings)
    model = MOIU.UniversalFallback(MOIU.Model{T}())
    return Optimizer{T}(
        settings,
        model,
        nothing,
        true,
        false,
        false,
        UInt64(0),
        0,
        MOI.OPTIMIZE_NOT_CALLED,
        MOI.NO_SOLUTION,
        MOI.NO_SOLUTION,
        STATUS_MESSAGES[QOCO_UNSOLVED],
    )
end

Optimizer(; kwargs...) = Optimizer{Float64}(; kwargs...)

@inline _function_kind(::Type{MOI.VariableIndex}) = UInt8(1)
@inline _function_kind(::Type{<:MOI.ScalarAffineFunction}) = UInt8(2)
@inline _function_kind(::Type{MOI.VectorOfVariables}) = UInt8(3)
@inline _function_kind(::Type{<:MOI.VectorAffineFunction}) = UInt8(4)
@inline _set_kind(::Type{<:MOI.EqualTo}) = UInt8(1)
@inline _set_kind(::Type{<:MOI.LessThan}) = UInt8(2)
@inline _set_kind(::Type{<:MOI.GreaterThan}) = UInt8(3)
@inline _set_kind(::Type{<:MOI.Interval}) = UInt8(4)
@inline _set_kind(::Type{MOI.Zeros}) = UInt8(5)
@inline _set_kind(::Type{MOI.Nonnegatives}) = UInt8(6)
@inline _set_kind(::Type{MOI.SecondOrderCone}) = UInt8(7)

@inline function _constraint_key(ci::MOI.ConstraintIndex{F,S}) where {F,S}
    return ConstraintKey(_function_kind(F), _set_kind(S), ci.value)
end

function _invalidate_results!(opt::Optimizer)
    opt.termination_status = MOI.OPTIMIZE_NOT_CALLED
    opt.primal_status = MOI.NO_SOLUTION
    opt.dual_status = MOI.NO_SOLUTION
    opt.raw_status_string = STATUS_MESSAGES[QOCO_UNSOLVED]
    return nothing
end

function _mark_structure_dirty!(opt::Optimizer)
    opt.structure_dirty = true
    opt.structure_generation += UInt64(1)
    _invalidate_results!(opt)
    return nothing
end

function _mark_numeric_dirty!(opt::Optimizer)
    _invalidate_results!(opt)
    return nothing
end

function _reset_optimizer!(opt::Optimizer{T}) where {T}
    opt.cache = nothing
    opt.structure_dirty = true
    opt.settings_dirty = false
    opt.starts_dirty = false
    opt.structure_generation += UInt64(1)
    opt.rebuild_count = 0
    _invalidate_results!(opt)
    return nothing
end

function MOI.empty!(opt::Optimizer)
    MOI.empty!(opt.model)
    _reset_optimizer!(opt)
    return nothing
end

MOI.is_empty(opt::Optimizer) = MOI.is_empty(opt.model)
MOI.supports_incremental_interface(::Optimizer) = true
Base.summary(io::IO, ::Optimizer{T}) where {T} = print(io, "JuliaQOCO.Optimizer{$T}")

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    index_map = MOI.copy_to(dest.model, src)
    _mark_structure_dirty!(dest)
    return index_map
end

# Incremental model construction. The UniversalFallback is only the editable
# MOI representation; numerical assembly is cached separately after solve one.
function MOI.add_variable(opt::Optimizer)
    vi = MOI.add_variable(opt.model)
    _mark_structure_dirty!(opt)
    return vi
end

function MOI.add_variables(opt::Optimizer, number::Int)
    variables = MOI.add_variables(opt.model, number)
    _mark_structure_dirty!(opt)
    return variables
end

function MOI.add_constraint(
    opt::Optimizer,
    function_value::F,
    set::S,
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    ci = MOI.add_constraint(opt.model, function_value, set)
    _mark_structure_dirty!(opt)
    return ci
end

function MOI.add_constraints(
    opt::Optimizer,
    functions::Vector{F},
    sets::Vector{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    constraints = MOI.add_constraints(opt.model, functions, sets)
    _mark_structure_dirty!(opt)
    return constraints
end

function MOI.add_constrained_variable(
    opt::Optimizer,
    set::S,
) where {S<:MOI.AbstractScalarSet}
    vi, ci = MOI.add_constrained_variable(opt.model, set)
    _mark_structure_dirty!(opt)
    return vi, ci
end

function MOI.add_constrained_variables(
    opt::Optimizer,
    set::S,
) where {S<:MOI.AbstractVectorSet}
    variables, ci = MOI.add_constrained_variables(opt.model, set)
    _mark_structure_dirty!(opt)
    return variables, ci
end

function MOI.delete(opt::Optimizer, index::MOI.Index)
    MOI.delete(opt.model, index)
    _mark_structure_dirty!(opt)
    return nothing
end

function MOI.delete(opt::Optimizer, indices::Vector{<:MOI.Index})
    MOI.delete(opt.model, indices)
    _mark_structure_dirty!(opt)
    return nothing
end

MOI.is_valid(opt::Optimizer, vi::MOI.VariableIndex) = MOI.is_valid(opt.model, vi)
MOI.is_valid(opt::Optimizer, ci::MOI.ConstraintIndex) = MOI.is_valid(opt.model, ci)

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.VariableIndex}) where {T} = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}) where {T} = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}) where {T} = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true
MOI.supports(::Optimizer, ::MOI.SolverVersion) = true
MOI.supports(::Optimizer, ::MOI.BarrierIterations) = true
MOI.supports(::Optimizer, ::MOI.DualObjectiveValue) = true
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart) = true
MOI.supports(opt::Optimizer, attr::MOI.AbstractVariableAttribute) =
    MOI.supports(opt.model, attr)
MOI.supports(opt::Optimizer, attr::MOI.AbstractConstraintAttribute) =
    MOI.supports(opt.model, attr)

MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VariableIndex}, ::Type{MOI.EqualTo{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VariableIndex}, ::Type{MOI.LessThan{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VariableIndex}, ::Type{MOI.GreaterThan{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VariableIndex}, ::Type{MOI.Interval{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{MOI.EqualTo{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{MOI.LessThan{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{MOI.GreaterThan{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{MOI.Interval{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorAffineFunction{T}}, ::Type{MOI.Zeros}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorAffineFunction{T}}, ::Type{MOI.Nonnegatives}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VectorAffineFunction{T}}, ::Type{MOI.SecondOrderCone}) where {T} = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{MOI.Zeros}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{MOI.Nonnegatives}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{MOI.SecondOrderCone}) = true

MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool) = (opt.settings.verbose = !value)
MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.settings.verbose

function MOI.set(opt::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    name = Symbol(attr.name)
    if name in fieldnames(typeof(opt.settings))
        converted = convert(fieldtype(typeof(opt.settings), name), value)
        old_value = getfield(opt.settings, name)
        setfield!(opt.settings, name, converted)
        if converted != old_value
            opt.settings_dirty = true
            if name in (
                :scaling_mode,
                :ruiz_iters,
                :kkt_static_reg,
                :kkt_dynamic_reg,
                :kkt_backend,
                :generated_max_dimension,
                :generated_max_factor_nnz,
            )
                _mark_structure_dirty!(opt)
            else
                _mark_numeric_dirty!(opt)
            end
        end
        return
    end
    throw(ArgumentError("Unknown JuliaQOCO setting $(attr.name)"))
end

function MOI.get(opt::Optimizer, attr::MOI.RawOptimizerAttribute)
    if attr.name == "structure_generation"
        return opt.structure_generation
    elseif attr.name == "rebuild_count"
        return opt.rebuild_count
    elseif attr.name == "last_commit_time_sec"
        return opt.cache === nothing ? 0.0 : opt.cache.last_commit_time_sec
    elseif attr.name == "active_kkt_backend"
        return opt.cache === nothing ? :none : active_kkt_backend(opt.cache.solver)
    end
    name = Symbol(attr.name)
    if name in fieldnames(typeof(opt.settings))
        return getfield(opt.settings, name)
    end
    throw(ArgumentError("Unknown JuliaQOCO setting $(attr.name)"))
end

MOI.get(::Optimizer, ::MOI.SolverName) = "JuliaQOCO"
MOI.get(::Optimizer, ::MOI.SolverVersion) = string(pkgversion(@__MODULE__))
MOI.get(opt::Optimizer, ::MOI.BarrierIterations) =
    opt.cache === nothing ? 0 : opt.cache.solver.solution.iters
MOI.get(opt::Optimizer, ::MOI.RawSolver) =
    opt.cache === nothing ? nothing : opt.cache.solver
MOI.get(opt::Optimizer, ::MOI.RawStatusString) = opt.raw_status_string
MOI.get(opt::Optimizer, ::MOI.TerminationStatus) = opt.termination_status
MOI.get(opt::Optimizer, ::MOI.PrimalStatus) = opt.primal_status
MOI.get(opt::Optimizer, ::MOI.DualStatus) = opt.dual_status
MOI.get(opt::Optimizer, ::MOI.ResultCount) =
    opt.primal_status == MOI.NO_SOLUTION ? 0 : 1
MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) =
    opt.cache === nothing ? 0.0 : opt.cache.solver.solution.solve_time_sec
MOI.get(opt::Optimizer, attr::MOI.DualObjectiveValue) =
    MOI.get(opt, MOI.ObjectiveValue(attr.result_index))

MOI.get(opt::Optimizer, attr::MOI.AbstractModelAttribute) = MOI.get(opt.model, attr)
MOI.get(opt::Optimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex) =
    MOI.get(opt.model, attr, vi)
MOI.get(opt::Optimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex) =
    MOI.get(opt.model, attr, ci)

function MOI.set(opt::Optimizer, attr::MOI.ObjectiveSense, value)
    MOI.set(opt.model, attr, value)
    _mark_structure_dirty!(opt)
    return nothing
end

function MOI.set(opt::Optimizer, attr::MOI.ObjectiveFunction{F}, value::F) where {F}
    MOI.set(opt.model, attr, value)
    _mark_structure_dirty!(opt)
    return nothing
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value,
)
    MOI.set(opt.model, attr, vi, value)
    opt.starts_dirty = true
    _mark_numeric_dirty!(opt)
    return nothing
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value,
)
    MOI.set(opt.model, attr, vi, value)
    return nothing
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex,
    value,
)
    MOI.set(opt.model, attr, ci, value)
    _mark_structure_dirty!(opt)
    return nothing
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
    value,
)
    MOI.set(opt.model, attr, ci, value)
    return nothing
end

function _constraint_value(x::AbstractVector{T}, f::MOI.VectorOfVariables) where {T<:AbstractFloat}
    return T[x[v.value] for v in f.variables]
end

function _constraint_value(
    x::AbstractVector{T},
    f::MOI.VectorOfVariables,
    variable_to_column::Vector{Int},
) where {T<:AbstractFloat}
    return T[x[variable_to_column[v.value]] for v in f.variables]
end

function _constraint_value(x::AbstractVector{T}, f::MOI.VectorAffineFunction{T}) where {T<:AbstractFloat}
    y = copy(f.constants)
    @inbounds for term in f.terms
        y[term.output_index] +=
            term.scalar_term.coefficient * x[term.scalar_term.variable.value]
    end
    return y
end

function _constraint_value(
    x::AbstractVector{T},
    f::MOI.VectorAffineFunction{T},
    variable_to_column::Vector{Int},
) where {T<:AbstractFloat}
    y = copy(f.constants)
    @inbounds for term in f.terms
        y[term.output_index] += term.scalar_term.coefficient *
                                x[variable_to_column[term.scalar_term.variable.value]]
    end
    return y
end

_constraint_value(x::AbstractVector, f::MOI.VariableIndex) = x[f.value]
_constraint_value(
    x::AbstractVector,
    f::MOI.VariableIndex,
    variable_to_column::Vector{Int},
) = x[variable_to_column[f.value]]

function _constraint_value(x::AbstractVector{T}, f::MOI.ScalarAffineFunction{T}) where {T<:AbstractFloat}
    y = f.constant
    @inbounds for term in f.terms
        y += term.coefficient * x[term.variable.value]
    end
    return y
end

function _constraint_value(
    x::AbstractVector{T},
    f::MOI.ScalarAffineFunction{T},
    variable_to_column::Vector{Int},
) where {T<:AbstractFloat}
    y = f.constant
    @inbounds for term in f.terms
        y += term.coefficient * x[variable_to_column[term.variable.value]]
    end
    return y
end

function _objective_data(
    model::MOIU.UniversalFallback{MOIU.Model{T}},
    variable_to_column::Vector{Int},
    n::Int,
) where {T<:AbstractFloat}
    rows = Int[]
    cols = Int[]
    vals = T[]
    c = zeros(T, n)
    constant = zero(T)
    pending = Dict{QuadraticKey,PendingTarget{T}}()
    sense = MOI.get(model, MOI.ObjectiveSense())
    sign = sense == MOI.MAX_SENSE ? -one(T) : one(T)
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    if F == MOI.VariableIndex
        vi = MOI.get(model, MOI.ObjectiveFunction{MOI.VariableIndex}())
        c[variable_to_column[vi.value]] = sign
    elseif F == MOI.ScalarAffineFunction{T}
        f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
        constant = f.constant
        @inbounds for term in f.terms
            c[variable_to_column[term.variable.value]] += sign * term.coefficient
        end
    elseif F == MOI.ScalarQuadraticFunction{T}
        f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}())
        constant = f.constant
        @inbounds for term in f.affine_terms
            c[variable_to_column[term.variable.value]] += sign * term.coefficient
        end
        @inbounds for term in f.quadratic_terms
            i = term.variable_1.value
            j = term.variable_2.value
            row, col = minmax(variable_to_column[i], variable_to_column[j])
            push!(rows, row)
            push!(cols, col)
            push!(vals, sign * term.coefficient)
            key_row, key_col = minmax(i, j)
            pending[QuadraticKey(key_row, key_col)] = PendingTarget{T}(
                _MATRIX_P,
                row,
                col,
                sign,
                0,
                0,
                zero(T),
            )
        end
    end
    return sparse(rows, cols, vals, n, n), c, constant, sign, pending
end

@inline function _register_target!(
    pending::Dict{CoefficientKey,PendingTarget{T}},
    key::CoefficientKey,
    matrix::UInt8,
    row::Int,
    column::Int,
    multiplier::T,
) where {T}
    pending[key] = PendingTarget{T}(
        matrix,
        row,
        column,
        multiplier,
        0,
        0,
        zero(T),
    )
    return nothing
end

function _register_interval_target!(
    pending::Dict{CoefficientKey,PendingTarget{T}},
    key::CoefficientKey,
    lower_row::Int,
    upper_row::Int,
    column::Int,
) where {T}
    pending[key] = PendingTarget{T}(
        _MATRIX_G,
        lower_row,
        column,
        -one(T),
        upper_row,
        column,
        one(T),
    )
    return nothing
end

function _constraint_data(
    opt::Optimizer{T},
    variable_to_column::Vector{Int},
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
    q = Int[]
    info = Dict{ConstraintKey,ConstraintInfo{T}}()
    pending = Dict{CoefficientKey,PendingTarget{T}}()
    eq_offset = 0
    cone_offset = 0

    # Equalities.
    for ci in MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T},MOI.EqualTo{T}}())
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        set = MOI.get(opt.model, MOI.ConstraintSet(), ci)
        push!(b, set.value - f.constant)
        key = _constraint_key(ci)
        @inbounds for term in f.terms
            push!(Arows, eq_offset + 1)
            push!(Acols, variable_to_column[term.variable.value])
            push!(Avals, term.coefficient)
            _register_target!(
                pending,
                CoefficientKey(key, 1, term.variable.value),
                _MATRIX_A,
                eq_offset + 1,
                variable_to_column[term.variable.value],
                one(T),
            )
        end
        info[key] = ConstraintInfo{T}(_ROW_EQUALITY, eq_offset + 1, 1, -one(T))
        eq_offset += 1
    end
    for ci in MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.EqualTo{T}}())
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        set = MOI.get(opt.model, MOI.ConstraintSet(), ci)
        push!(Arows, eq_offset + 1)
        push!(Acols, variable_to_column[f.value])
        push!(Avals, one(T))
        push!(b, set.value)
        info[_constraint_key(ci)] =
            ConstraintInfo{T}(_ROW_EQUALITY, eq_offset + 1, 1, -one(T))
        eq_offset += 1
    end
    for ci in MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T},MOI.Zeros}())
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.constants)
        append!(b, -f.constants)
        key = _constraint_key(ci)
        @inbounds for term in f.terms
            row = eq_offset + term.output_index
            col = variable_to_column[term.scalar_term.variable.value]
            push!(Arows, row)
            push!(Acols, col)
            push!(Avals, term.scalar_term.coefficient)
            _register_target!(
                pending,
                CoefficientKey(key, term.output_index, term.scalar_term.variable.value),
                _MATRIX_A,
                row,
                col,
                one(T),
            )
        end
        info[key] =
            ConstraintInfo{T}(_ROW_EQUALITY, eq_offset + 1, dim, -one(T))
        eq_offset += dim
    end
    for ci in MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MOI.Zeros}())
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.variables)
        append!(b, zeros(T, dim))
        @inbounds for (i, vi) in enumerate(f.variables)
            push!(Arows, eq_offset + i)
            push!(Acols, variable_to_column[vi.value])
            push!(Avals, one(T))
        end
        info[_constraint_key(ci)] =
            ConstraintInfo{T}(_ROW_EQUALITY, eq_offset + 1, dim, -one(T))
        eq_offset += dim
    end

    # Scalar affine inequalities and variable bounds become orthant rows.
    for F in (MOI.ScalarAffineFunction{T}, MOI.VariableIndex)
        for S in (MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T})
            for ci in MOI.get(opt.model, MOI.ListOfConstraintIndices{F,S}())
                f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
                set = MOI.get(opt.model, MOI.ConstraintSet(), ci)
                constant = f isa MOI.VariableIndex ? zero(T) : f.constant
                terms = f isa MOI.VariableIndex ?
                        (MOI.ScalarAffineTerm(one(T), f),) : f.terms
                key = _constraint_key(ci)
                if set isa MOI.GreaterThan{T}
                    if !isfinite(set.lower)
                        info[key] = ConstraintInfo{T}(_ROW_CONE, 0, 0, one(T))
                        continue
                    end
                    push!(h, constant - set.lower)
                    @inbounds for term in terms
                        push!(Grows, cone_offset + 1)
                        push!(Gcols, variable_to_column[term.variable.value])
                        push!(Gvals, -term.coefficient)
                        f isa MOI.VariableIndex || _register_target!(
                            pending,
                            CoefficientKey(key, 1, term.variable.value),
                            _MATRIX_G,
                            cone_offset + 1,
                            variable_to_column[term.variable.value],
                            -one(T),
                        )
                    end
                    info[key] =
                        ConstraintInfo{T}(_ROW_CONE, cone_offset + 1, 1, one(T))
                    cone_offset += 1
                elseif set isa MOI.LessThan{T}
                    if !isfinite(set.upper)
                        info[key] =
                            ConstraintInfo{T}(_ROW_CONE, 0, 0, -one(T))
                        continue
                    end
                    push!(h, set.upper - constant)
                    @inbounds for term in terms
                        push!(Grows, cone_offset + 1)
                        push!(Gcols, variable_to_column[term.variable.value])
                        push!(Gvals, term.coefficient)
                        f isa MOI.VariableIndex || _register_target!(
                            pending,
                            CoefficientKey(key, 1, term.variable.value),
                            _MATRIX_G,
                            cone_offset + 1,
                            variable_to_column[term.variable.value],
                            one(T),
                        )
                    end
                    info[key] =
                        ConstraintInfo{T}(_ROW_CONE, cone_offset + 1, 1, -one(T))
                    cone_offset += 1
                else
                    finite_lower = isfinite(set.lower)
                    finite_upper = isfinite(set.upper)
                    if !finite_lower && !finite_upper
                        info[key] =
                            ConstraintInfo{T}(_ROW_CONE, 0, 0, one(T))
                    elseif finite_lower && finite_upper
                        push!(h, constant - set.lower)
                        push!(h, set.upper - constant)
                        @inbounds for term in terms
                            col = variable_to_column[term.variable.value]
                            push!(Grows, cone_offset + 1)
                            push!(Gcols, col)
                            push!(Gvals, -term.coefficient)
                            push!(Grows, cone_offset + 2)
                            push!(Gcols, col)
                            push!(Gvals, term.coefficient)
                            f isa MOI.VariableIndex ||
                                _register_interval_target!(
                                    pending,
                                    CoefficientKey(key, 1, term.variable.value),
                                    cone_offset + 1,
                                    cone_offset + 2,
                                    col,
                                )
                        end
                        info[key] = ConstraintInfo{T}(
                            _ROW_CONE,
                            cone_offset + 1,
                            2,
                            one(T),
                        )
                        cone_offset += 2
                    else
                        is_lower = finite_lower
                        push!(
                            h,
                            is_lower ? constant - set.lower :
                            set.upper - constant,
                        )
                        multiplier = is_lower ? -one(T) : one(T)
                        @inbounds for term in terms
                            col = variable_to_column[term.variable.value]
                            push!(Grows, cone_offset + 1)
                            push!(Gcols, col)
                            push!(Gvals, multiplier * term.coefficient)
                            f isa MOI.VariableIndex || _register_target!(
                                pending,
                                CoefficientKey(key, 1, term.variable.value),
                                _MATRIX_G,
                                cone_offset + 1,
                                col,
                                multiplier,
                            )
                        end
                        info[key] = ConstraintInfo{T}(
                            _ROW_CONE,
                            cone_offset + 1,
                            1,
                            is_lower ? one(T) : -one(T),
                        )
                        cone_offset += 1
                    end
                end
            end
        end
    end

    for ci in MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T},MOI.Nonnegatives}())
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.constants)
        append!(h, f.constants)
        key = _constraint_key(ci)
        @inbounds for term in f.terms
            row = cone_offset + term.output_index
            col = variable_to_column[term.scalar_term.variable.value]
            push!(Grows, row)
            push!(Gcols, col)
            push!(Gvals, -term.scalar_term.coefficient)
            _register_target!(
                pending,
                CoefficientKey(key, term.output_index, term.scalar_term.variable.value),
                _MATRIX_G,
                row,
                col,
                -one(T),
            )
        end
        info[key] = ConstraintInfo{T}(_ROW_CONE, cone_offset + 1, dim, one(T))
        cone_offset += dim
    end
    for ci in MOI.get(opt.model, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,MOI.Nonnegatives}())
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        dim = length(f.variables)
        append!(h, zeros(T, dim))
        @inbounds for (i, vi) in enumerate(f.variables)
            push!(Grows, cone_offset + i)
            push!(Gcols, variable_to_column[vi.value])
            push!(Gvals, -one(T))
        end
        info[_constraint_key(ci)] =
            ConstraintInfo{T}(_ROW_CONE, cone_offset + 1, dim, one(T))
        cone_offset += dim
    end
    l = cone_offset

    for F in (MOI.VectorAffineFunction{T}, MOI.VectorOfVariables)
        for ci in MOI.get(opt.model, MOI.ListOfConstraintIndices{F,MOI.SecondOrderCone}())
            f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
            dim = MOI.dimension(MOI.get(opt.model, MOI.ConstraintSet(), ci))
            key = _constraint_key(ci)
            if f isa MOI.VectorAffineFunction{T}
                append!(h, f.constants)
                @inbounds for term in f.terms
                    row = cone_offset + term.output_index
                    col = variable_to_column[term.scalar_term.variable.value]
                    push!(Grows, row)
                    push!(Gcols, col)
                    push!(Gvals, -term.scalar_term.coefficient)
                    _register_target!(
                        pending,
                        CoefficientKey(key, term.output_index, term.scalar_term.variable.value),
                        _MATRIX_G,
                        row,
                        col,
                        -one(T),
                    )
                end
            else
                append!(h, zeros(T, dim))
                @inbounds for (i, vi) in enumerate(f.variables)
                    push!(Grows, cone_offset + i)
                    push!(Gcols, variable_to_column[vi.value])
                    push!(Gvals, -one(T))
                end
            end
            info[key] =
                ConstraintInfo{T}(_ROW_CONE, cone_offset + 1, dim, one(T))
            cone_offset += dim
            push!(q, dim)
        end
    end

    supported = Set([
        (MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}),
        (MOI.ScalarAffineFunction{T}, MOI.LessThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.Interval{T}),
        (MOI.VariableIndex, MOI.EqualTo{T}),
        (MOI.VariableIndex, MOI.LessThan{T}),
        (MOI.VariableIndex, MOI.GreaterThan{T}),
        (MOI.VariableIndex, MOI.Interval{T}),
        (MOI.VectorAffineFunction{T}, MOI.Zeros),
        (MOI.VectorAffineFunction{T}, MOI.Nonnegatives),
        (MOI.VectorAffineFunction{T}, MOI.SecondOrderCone),
        (MOI.VectorOfVariables, MOI.Zeros),
        (MOI.VectorOfVariables, MOI.Nonnegatives),
        (MOI.VectorOfVariables, MOI.SecondOrderCone),
    ])
    for type_pair in MOI.get(opt.model, MOI.ListOfConstraintTypesPresent())
        type_pair in supported && continue
        cis = MOI.get(
            opt.model,
            MOI.ListOfConstraintIndices{type_pair[1],type_pair[2]}(),
        )
        isempty(cis) || throw(
            ArgumentError(
                "Unsupported constraint type $(type_pair[1]) in $(type_pair[2]). " *
                "Use JuMP.Model with bridges for this form.",
            ),
        )
    end

    A = sparse(Arows, Acols, Avals, eq_offset, n)
    G = sparse(Grows, Gcols, Gvals, cone_offset, n)
    return A, b, G, h, l, q, info, pending
end

function _coordinate_positions(A::SparseMatrixCSC)
    positions = Dict{Tuple{Int,Int},Int}()
    @inbounds for column in 1:size(A, 2)
        for position in A.colptr[column]:(A.colptr[column + 1] - 1)
            positions[(A.rowval[position], column)] = position
        end
    end
    return positions
end

function _finalize_target(
    pending::PendingTarget{T},
    raw_positions::NTuple{3,Dict{Tuple{Int,Int},Int}},
    native_positions::NTuple{3,Dict{Tuple{Int,Int},Int}},
) where {T}
    matrix = Int(pending.matrix)
    coordinate_1 = (pending.row_1, pending.column_1)
    raw_1 = raw_positions[matrix][coordinate_1]
    native_1 = native_positions[matrix][coordinate_1]
    if pending.row_2 == 0
        return MatrixTarget{T}(
            pending.matrix,
            native_1,
            raw_1,
            pending.multiplier_1,
            0,
            0,
            zero(T),
        )
    end
    coordinate_2 = (pending.row_2, pending.column_2)
    return MatrixTarget{T}(
        pending.matrix,
        native_1,
        raw_1,
        pending.multiplier_1,
        native_positions[matrix][coordinate_2],
        raw_positions[matrix][coordinate_2],
        pending.multiplier_2,
    )
end

function _build_cache!(opt::Optimizer{T}) where {T<:AbstractFloat}
    variables = MOI.get(opt.model, MOI.ListOfVariableIndices())
    max_variable = isempty(variables) ? 0 : maximum(vi.value for vi in variables)
    variable_to_column = zeros(Int, max_variable)
    @inbounds for (column, vi) in enumerate(variables)
        variable_to_column[vi.value] = column
    end
    n = length(variables)
    P, c, constant, sign, pending_quadratic =
        _objective_data(opt.model, variable_to_column, n)
    A, b, G, h, l, q, info, pending_coefficients =
        _constraint_data(opt, variable_to_column, n)
    solver = Solver(P, c, A, b, G, h, l, q; settings = copy_settings(opt.settings))
    raw_positions = (
        _coordinate_positions(P),
        _coordinate_positions(A),
        _coordinate_positions(G),
    )
    native_positions = (
        _coordinate_positions(solver.data.P),
        _coordinate_positions(solver.data.A),
        _coordinate_positions(solver.data.G),
    )
    coefficient_targets = Dict{CoefficientKey,MatrixTarget{T}}()
    for (key, pending) in pending_coefficients
        coefficient_targets[key] =
            _finalize_target(pending, raw_positions, native_positions)
    end
    quadratic_targets = Dict{QuadraticKey,MatrixTarget{T}}()
    for (key, pending) in pending_quadratic
        quadratic_targets[key] =
            _finalize_target(pending, raw_positions, native_positions)
    end
    P_values_by_native = zeros(T, nnz(solver.data.P))
    @inbounds for target in values(quadratic_targets)
        P_values_by_native[target.position_1] = P.nzval[target.raw_position_1]
    end
    opt.rebuild_count += 1
    opt.cache = MOIAssemblyCache{T}(
        solver,
        info,
        coefficient_targets,
        quadratic_targets,
        copy(P.nzval),
        copy(A.nzval),
        copy(G.nzval),
        DirtyQueue(P_values_by_native),
        DirtyQueue(copy(A.nzval)),
        DirtyQueue(copy(G.nzval)),
        DirtyQueue(copy(c)),
        DirtyQueue(copy(b)),
        DirtyQueue(copy(h)),
        variable_to_column,
        variables,
        zeros(T, length(c)),
        constant,
        sign,
        opt.structure_generation,
        opt.rebuild_count,
        0,
        0.0,
    )
    opt.structure_dirty = false
    opt.settings_dirty = false
    return opt.cache
end

@inline function _queue_target!(
    cache::MOIAssemblyCache{T},
    target::MatrixTarget{T},
    coefficient::T,
) where {T}
    if target.matrix == _MATRIX_P
        value = target.multiplier_1 * coefficient
        cache.raw_P[target.raw_position_1] = value
        _queue!(cache.P_dirty, target.position_1, value)
        if target.position_2 != 0
            value_2 = target.multiplier_2 * coefficient
            cache.raw_P[target.raw_position_2] = value_2
            _queue!(cache.P_dirty, target.position_2, value_2)
        end
    elseif target.matrix == _MATRIX_A
        value = target.multiplier_1 * coefficient
        cache.raw_A[target.raw_position_1] = value
        _queue!(cache.A_dirty, target.position_1, value)
        if target.position_2 != 0
            value_2 = target.multiplier_2 * coefficient
            cache.raw_A[target.raw_position_2] = value_2
            _queue!(cache.A_dirty, target.position_2, value_2)
        end
    else
        value = target.multiplier_1 * coefficient
        cache.raw_G[target.raw_position_1] = value
        _queue!(cache.G_dirty, target.position_1, value)
        if target.position_2 != 0
            value_2 = target.multiplier_2 * coefficient
            cache.raw_G[target.raw_position_2] = value_2
            _queue!(cache.G_dirty, target.position_2, value_2)
        end
    end
    return nothing
end

function _queue_constraint_coefficient!(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex,
    output_index::Int,
    variable::MOI.VariableIndex,
    coefficient::T,
) where {T}
    cache = opt.cache
    (cache === nothing || opt.structure_dirty) && return nothing
    key = CoefficientKey(_constraint_key(ci), output_index, variable.value)
    target = get(cache.coefficient_targets, key, nothing)
    if target === nothing
        _mark_structure_dirty!(opt)
    else
        _queue_target!(cache, target, coefficient)
        _mark_numeric_dirty!(opt)
    end
    return nothing
end

function MOI.modify(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex,
    change::MOI.ScalarCoefficientChange{T},
) where {T}
    MOI.modify(opt.model, ci, change)
    _queue_constraint_coefficient!(
        opt,
        ci,
        1,
        change.variable,
        change.new_coefficient,
    )
    return nothing
end

function MOI.modify(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex,
    change::MOI.MultirowChange{T},
) where {T}
    MOI.modify(opt.model, ci, change)
    @inbounds for (output_index, coefficient) in change.new_coefficients
        _queue_constraint_coefficient!(
            opt,
            ci,
            output_index,
            change.variable,
            coefficient,
        )
    end
    return nothing
end

function MOI.modify(
    opt::Optimizer{T},
    attr::MOI.ObjectiveFunction,
    change::MOI.ScalarCoefficientChange{T},
) where {T}
    MOI.modify(opt.model, attr, change)
    cache = opt.cache
    if cache !== nothing && !opt.structure_dirty
        _queue!(
            cache.c_dirty,
            cache.variable_to_column[change.variable.value],
            cache.objective_sign * change.new_coefficient,
        )
        _mark_numeric_dirty!(opt)
    end
    return nothing
end

function MOI.modify(
    opt::Optimizer{T},
    attr::MOI.ObjectiveFunction,
    change::MOI.ScalarQuadraticCoefficientChange{T},
) where {T}
    MOI.modify(opt.model, attr, change)
    cache = opt.cache
    if cache !== nothing && !opt.structure_dirty
        row, column = minmax(change.variable_1.value, change.variable_2.value)
        target = get(cache.quadratic_targets, QuadraticKey(row, column), nothing)
        if target === nothing
            _mark_structure_dirty!(opt)
        else
            _queue_target!(cache, target, change.new_coefficient)
            _mark_numeric_dirty!(opt)
        end
    end
    return nothing
end

function MOI.modify(
    opt::Optimizer{T},
    attr::MOI.ObjectiveFunction,
    change::MOI.ScalarConstantChange{T},
) where {T}
    MOI.modify(opt.model, attr, change)
    opt.cache === nothing || (opt.cache.objective_constant = change.new_constant)
    _mark_numeric_dirty!(opt)
    return nothing
end

function _queue_constraint_constants!(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex{F,S},
) where {T,F,S}
    cache = opt.cache
    (cache === nothing || opt.structure_dirty) && return nothing
    info = cache.constraint_info[_constraint_key(ci)]
    info.length == 0 && return nothing
    f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
    set = MOI.get(opt.model, MOI.ConstraintSet(), ci)
    if f isa MOI.VectorAffineFunction{T}
        @inbounds for i in eachindex(f.constants)
            value = S == MOI.Zeros ? -f.constants[i] : f.constants[i]
            queue = S == MOI.Zeros ? cache.b_dirty : cache.h_dirty
            _queue!(queue, info.offset + i - 1, value)
        end
    else
        constant = f isa MOI.VariableIndex ? zero(T) : f.constant
        if set isa MOI.EqualTo{T}
            _queue!(cache.b_dirty, info.offset, set.value - constant)
        elseif set isa MOI.GreaterThan{T}
            _queue!(cache.h_dirty, info.offset, constant - set.lower)
        elseif set isa MOI.LessThan{T}
            _queue!(cache.h_dirty, info.offset, set.upper - constant)
        elseif set isa MOI.Interval{T}
            if info.length == 2
                _queue!(cache.h_dirty, info.offset, constant - set.lower)
                _queue!(
                    cache.h_dirty,
                    info.offset + 1,
                    set.upper - constant,
                )
            elseif info.dual_sign > zero(T)
                _queue!(cache.h_dirty, info.offset, constant - set.lower)
            else
                _queue!(cache.h_dirty, info.offset, set.upper - constant)
            end
        end
    end
    _mark_numeric_dirty!(opt)
    return nothing
end

function MOI.modify(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex,
    change::MOI.ScalarConstantChange{T},
) where {T}
    MOI.modify(opt.model, ci, change)
    _queue_constraint_constants!(opt, ci)
    return nothing
end

function MOI.modify(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex,
    change::MOI.VectorConstantChange{T},
) where {T}
    MOI.modify(opt.model, ci, change)
    _queue_constraint_constants!(opt, ci)
    return nothing
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex,
    set,
)
    old_set = MOI.get(opt.model, attr, ci)
    MOI.set(opt.model, attr, ci, set)
    if _finite_pattern(old_set) == _finite_pattern(set)
        _queue_constraint_constants!(opt, ci)
    else
        _mark_structure_dirty!(opt)
    end
    return nothing
end

_finite_pattern(::MOI.AbstractSet) = UInt8(0)
_finite_pattern(set::MOI.EqualTo) = UInt8(1)
_finite_pattern(set::MOI.GreaterThan) = isfinite(set.lower) ? UInt8(2) : UInt8(3)
_finite_pattern(set::MOI.LessThan) = isfinite(set.upper) ? UInt8(4) : UInt8(5)
_finite_pattern(set::MOI.Interval) =
    UInt8((isfinite(set.lower) ? 1 : 0) + (isfinite(set.upper) ? 2 : 0) + 6)

function _commit_updates!(cache::MOIAssemblyCache{T}) where {T}
    t0 = time_ns()
    solver = cache.solver
    matrix_dirty =
        !isempty(cache.P_dirty.indices) ||
        !isempty(cache.A_dirty.indices) ||
        !isempty(cache.G_dirty.indices)
    if matrix_dirty && solver.settings.scaling_mode == :recompute
        update_matrix_data!(
            solver;
            Px = isempty(cache.P_dirty.indices) ? nothing : cache.raw_P,
            Ax = isempty(cache.A_dirty.indices) ? nothing : cache.raw_A,
            Gx = isempty(cache.G_dirty.indices) ? nothing : cache.raw_G,
        )
    elseif matrix_dirty
        isempty(cache.P_dirty.indices) || update_P_entries!(
            solver,
            cache.P_dirty.indices,
            _commit_values!(cache.P_dirty),
        )
        isempty(cache.A_dirty.indices) || update_A_entries!(
            solver,
            cache.A_dirty.indices,
            _commit_values!(cache.A_dirty),
        )
        isempty(cache.G_dirty.indices) || update_G_entries!(
            solver,
            cache.G_dirty.indices,
            _commit_values!(cache.G_dirty),
        )
    end
    isempty(cache.c_dirty.indices) || update_c_entries!(
        solver,
        cache.c_dirty.indices,
        _commit_values!(cache.c_dirty),
    )
    isempty(cache.b_dirty.indices) || update_b_entries!(
        solver,
        cache.b_dirty.indices,
        _commit_values!(cache.b_dirty),
    )
    isempty(cache.h_dirty.indices) || update_h_entries!(
        solver,
        cache.h_dirty.indices,
        _commit_values!(cache.h_dirty),
    )
    _clear_queue!(cache.P_dirty)
    _clear_queue!(cache.A_dirty)
    _clear_queue!(cache.G_dirty)
    _clear_queue!(cache.c_dirty)
    _clear_queue!(cache.b_dirty)
    _clear_queue!(cache.h_dirty)
    cache.commit_count += 1
    cache.last_commit_time_sec = elapsed_time_sec(t0)
    return nothing
end

@inline function _has_pending_updates(cache::MOIAssemblyCache)
    return !isempty(cache.P_dirty.indices) ||
           !isempty(cache.A_dirty.indices) ||
           !isempty(cache.G_dirty.indices) ||
           !isempty(cache.c_dirty.indices) ||
           !isempty(cache.b_dirty.indices) ||
           !isempty(cache.h_dirty.indices)
end

function _apply_variable_starts!(opt::Optimizer{T}) where {T}
    cache = opt.cache
    cache === nothing && return nothing
    solver = cache.solver
    if solver.solution.status == QOCO_UNSOLVED
        fill!(cache.start_buffer, zero(T))
    else
        copyto!(cache.start_buffer, solver.solution.x)
    end
    found = false
    @inbounds for index in eachindex(cache.start_buffer)
        value = MOI.get(
            opt.model,
            MOI.VariablePrimalStart(),
            cache.column_to_variable[index],
        )
        if value !== nothing
            cache.start_buffer[index] = value
            found = true
        end
    end
    found && warm_start!(solver; x = cache.start_buffer)
    opt.starts_dirty = false
    return nothing
end

function _copy_runtime_settings!(solver::Solver, settings::Settings)
    solver.settings = copy_settings(settings)
    return nothing
end

function _set_result_status!(opt::Optimizer)
    solver = opt.cache.solver
    opt.raw_status_string =
        status_string(solver.solution.status, solver.solution.status_detail)
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

function MOI.optimize!(opt::Optimizer)
    if opt.cache === nothing || opt.structure_dirty || !opt.settings.reuse_solver
        _build_cache!(opt)
    else
        if opt.settings_dirty
            _copy_runtime_settings!(opt.cache.solver, opt.settings)
            opt.settings_dirty = false
        end
        _has_pending_updates(opt.cache) && _commit_updates!(opt.cache)
    end
    opt.starts_dirty && _apply_variable_starts!(opt)
    solve!(opt.cache.solver)
    _set_result_status!(opt)
    return nothing
end

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    cache = opt.cache
    return cache.objective_sign * cache.solver.solution.obj +
           cache.objective_constant
end

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(opt, attr)
    return opt.cache.solver.solution.x[opt.cache.variable_to_column[vi.value]]
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    MOI.check_result_index_bounds(opt, attr)
    info = opt.cache.constraint_info[_constraint_key(ci)]
    source = info.row_kind == _ROW_EQUALITY ?
             opt.cache.solver.solution.y : opt.cache.solver.solution.z
    if info.length == 0
        return F <: MOI.AbstractVectorFunction ? eltype(source)[] :
               zero(eltype(source))
    end
    if S <: MOI.Interval && info.length == 2
        return source[info.offset] - source[info.offset + 1]
    end
    if F <: MOI.AbstractVectorFunction
        result = Vector{eltype(source)}(undef, info.length)
        @inbounds for i in 1:info.length
            result[i] = info.dual_sign * source[info.offset + i - 1]
        end
        return result
    end
    return info.dual_sign * source[info.offset]
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(opt, attr)
    x = opt.cache.solver.solution.x
    f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
    return _constraint_value(x, f, opt.cache.variable_to_column)
end
