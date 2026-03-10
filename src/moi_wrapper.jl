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
        STATUS_MESSAGES[QOCO_UNSOLVED],
    )
end

Optimizer(; kwargs...) = Optimizer{Float64}(; kwargs...)

function _reset_results!(opt::Optimizer{T}) where {T<:AbstractFloat}
    opt.native_solver = nothing
    empty!(opt.constraint_info)
    opt.objective_constant = zero(T)
    opt.objective_sign = one(T)
    opt.termination_status = MOI.OPTIMIZE_NOT_CALLED
    opt.primal_status = MOI.NO_SOLUTION
    opt.dual_status = MOI.NO_SOLUTION
    opt.raw_status_string = STATUS_MESSAGES[QOCO_UNSOLVED]
    return nothing
end

MOI.empty!(opt::Optimizer) = (MOI.empty!(opt.model); _reset_results!(opt); nothing)
MOI.is_empty(opt::Optimizer) = MOI.is_empty(opt.model)
MOI.supports_incremental_interface(::Optimizer) = false
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
MOI.get(opt::Optimizer, ::MOI.RawStatusString) = opt.raw_status_string
MOI.get(opt::Optimizer, ::MOI.TerminationStatus) = opt.termination_status
MOI.get(opt::Optimizer, ::MOI.PrimalStatus) = opt.primal_status
MOI.get(opt::Optimizer, ::MOI.DualStatus) = opt.dual_status
MOI.get(opt::Optimizer, ::MOI.ResultCount) = opt.primal_status == MOI.NO_SOLUTION ? 0 : 1
MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) = opt.native_solver === nothing ? 0.0 : opt.native_solver.solution.solve_time_sec

MOI.is_valid(opt::Optimizer, vi::MOI.VariableIndex) = MOI.is_valid(opt.model, vi)
MOI.is_valid(opt::Optimizer, ci::MOI.ConstraintIndex) = MOI.is_valid(opt.model, ci)

MOI.get(opt::Optimizer, attr::MOI.AbstractModelAttribute) = MOI.get(opt.model, attr)
MOI.get(opt::Optimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex) = MOI.get(opt.model, attr, vi)
MOI.get(opt::Optimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex) = MOI.get(opt.model, attr, ci)

function _constraint_value(x::AbstractVector{T}, f::MOI.VectorOfVariables) where {T<:AbstractFloat}
    return T[x[v.value] for v in f.variables]
end

function _constraint_value(x::AbstractVector{T}, f::MOI.VectorAffineFunction{T}) where {T<:AbstractFloat}
    y = copy(f.constants)
    for term in f.terms
        y[term.output_index] += term.scalar_term.coefficient * x[term.scalar_term.variable.value]
    end
    return y
end

function _constraint_value(x::AbstractVector{T}, f::MOI.VariableIndex) where {T<:AbstractFloat}
    return x[f.value]
end

function _constraint_value(x::AbstractVector{T}, f::MOI.ScalarAffineFunction{T}) where {T<:AbstractFloat}
    y = f.constant
    for term in f.terms
        y += term.coefficient * x[term.variable.value]
    end
    return y
end

function _objective_data(model::MOIU.UniversalFallback{MOIU.Model{T}}) where {T<:AbstractFloat}
    n = MOI.get(model, MOI.NumberOfVariables())
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
        c[vi.value] = sign
    elseif F == MOI.ScalarAffineFunction{T}
        f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
        constant = f.constant
        for term in f.terms
            c[term.variable.value] += sign * term.coefficient
        end
    elseif F == MOI.ScalarQuadraticFunction{T}
        f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}())
        constant = f.constant
        for term in f.affine_terms
            c[term.variable.value] += sign * term.coefficient
        end
        for term in f.quadratic_terms
            i = term.variable_1.value
            j = term.variable_2.value
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

function _append_vector_affine_eq!(
    Arows::Vector{Int},
    Acols::Vector{Int},
    Avals::Vector{T},
    b::Vector{T},
    f::MOI.VectorAffineFunction{T},
    row_offset::Int,
) where {T<:AbstractFloat}
    append!(b, -f.constants)
    for term in f.terms
        push!(Arows, row_offset + term.output_index)
        push!(Acols, term.scalar_term.variable.value)
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
) where {T<:AbstractFloat}
    append!(h, f.constants)
    for term in f.terms
        push!(Grows, row_offset + term.output_index)
        push!(Gcols, term.scalar_term.variable.value)
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
) where {T<:AbstractFloat}
    append!(b, zeros(T, length(f.variables)))
    for (i, vi) in enumerate(f.variables)
        push!(Arows, row_offset + i)
        push!(Acols, vi.value)
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
) where {T<:AbstractFloat}
    append!(h, zeros(T, length(f.variables)))
    for (i, vi) in enumerate(f.variables)
        push!(Grows, row_offset + i)
        push!(Gcols, vi.value)
        push!(Gvals, -one(T))
    end
    return nothing
end

function _constraint_data(opt::Optimizer{T}) where {T<:AbstractFloat}
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
    info = Dict{Any,ConstraintInfo{T}}()

    eq_offset = 0
    cone_offset = 0

    for (F, S) in MOI.get(opt.model, MOI.ListOfConstraintTypesPresent())
        cis = MOI.get(opt.model, MOI.ListOfConstraintIndices{F,S}())
        if F == MOI.VectorAffineFunction{T} && S == MOI.Zeros
            for ci in cis
                f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
                dim = length(f.constants)
                _append_vector_affine_eq!(Arows, Acols, Avals, b, f, eq_offset)
                info[ci] = ConstraintInfo{T}(kind = :eq, offset = eq_offset + 1, length = dim, sign = -one(T))
                eq_offset += dim
            end
        elseif F == MOI.VectorOfVariables && S == MOI.Zeros
            for ci in cis
                f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
                dim = length(f.variables)
                _append_vector_of_variables_eq!(Arows, Acols, Avals, b, f, eq_offset)
                info[ci] = ConstraintInfo{T}(kind = :eq, offset = eq_offset + 1, length = dim, sign = -one(T))
                eq_offset += dim
            end
        elseif F == MOI.VectorAffineFunction{T} && S == MOI.Nonnegatives
            for ci in cis
                f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
                dim = length(f.constants)
                _append_vector_affine_cone!(Grows, Gcols, Gvals, h, f, cone_offset)
                info[ci] = ConstraintInfo{T}(kind = :cone, offset = cone_offset + 1, length = dim)
                cone_offset += dim
                l += dim
            end
        elseif F == MOI.VectorOfVariables && S == MOI.Nonnegatives
            for ci in cis
                f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
                dim = length(f.variables)
                _append_vector_of_variables_cone!(Grows, Gcols, Gvals, h, f, cone_offset)
                info[ci] = ConstraintInfo{T}(kind = :cone, offset = cone_offset + 1, length = dim)
                cone_offset += dim
                l += dim
            end
        elseif F == MOI.VectorAffineFunction{T} && S == MOI.SecondOrderCone
            for ci in cis
                f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
                dim = length(f.constants)
                _append_vector_affine_cone!(Grows, Gcols, Gvals, h, f, cone_offset)
                info[ci] = ConstraintInfo{T}(kind = :cone, offset = cone_offset + 1, length = dim)
                cone_offset += dim
                push!(q, dim)
            end
        elseif F == MOI.VectorOfVariables && S == MOI.SecondOrderCone
            for ci in cis
                f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
                dim = length(f.variables)
                _append_vector_of_variables_cone!(Grows, Gcols, Gvals, h, f, cone_offset)
                info[ci] = ConstraintInfo{T}(kind = :cone, offset = cone_offset + 1, length = dim)
                cone_offset += dim
                push!(q, dim)
            end
        elseif !isempty(cis)
            throw(ArgumentError("Unsupported constraint type $(F) in $(S). Use the bridged optimizer interface."))
        end
    end

    n = MOI.get(opt.model, MOI.NumberOfVariables())
    A = sparse(Arows, Acols, Avals, eq_offset, n)
    G = sparse(Grows, Gcols, Gvals, cone_offset, n)
    return A, b, G, h, l, q, info
end

function MOI.optimize!(opt::Optimizer{T}) where {T<:AbstractFloat}
    _reset_results!(opt)
    P, c, constant, sign = _objective_data(opt.model)
    A, b, G, h, l, q, info = _constraint_data(opt)
    solver = solve(P, c, A, b, G, h, l, q; settings = copy_settings(opt.settings))
    opt.native_solver = solver
    opt.constraint_info = info
    opt.objective_constant = constant
    opt.objective_sign = sign
    opt.raw_status_string = STATUS_MESSAGES[solver.solution.status]

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
    return opt.native_solver.solution.x[vi.value]
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
    return _constraint_value(x, f)
end
