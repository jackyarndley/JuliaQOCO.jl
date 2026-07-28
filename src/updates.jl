function _refresh_static_kkt!(solver::Solver{T}) where {T<:AbstractFloat}
    solver.linsys.factor === nothing && return solver
    _fill_static_values!(solver.linsys.static_values, solver.data)
    QDLDL.update_values_internal!(solver.linsys.factor, solver.linsys.static2kkt, solver.linsys.static_values)
    return solver
end

function _copy_start_component!(
    dest::Vector{T},
    value,
    fallback::AbstractVector{T},
    name::AbstractString,
) where {T<:AbstractFloat}
    if value === nothing
        if length(fallback) == length(dest)
            copyto!(dest, fallback)
        else
            fill!(dest, zero(T))
        end
    else
        length(value) == length(dest) || throw(ArgumentError("Warm-start $name has incorrect length"))
        copyto!(dest, value)
    end
    return dest
end

function warm_start!(
    solver::Solver{T};
    x = nothing,
    s = nothing,
    y = nothing,
    z = nothing,
) where {T<:AbstractFloat}
    fallback_x = solver.solution.status == QOCO_UNSOLVED ? T[] : solver.solution.x
    fallback_s = solver.solution.status == QOCO_UNSOLVED ? T[] : solver.solution.s
    fallback_y = solver.solution.status == QOCO_UNSOLVED ? T[] : solver.solution.y
    fallback_z = solver.solution.status == QOCO_UNSOLVED ? T[] : solver.solution.z
    _copy_start_component!(solver.warmstart.x, x, fallback_x, "x")
    _copy_start_component!(solver.warmstart.s, s, fallback_s, "s")
    _copy_start_component!(solver.warmstart.y, y, fallback_y, "y")
    _copy_start_component!(solver.warmstart.z, z, fallback_z, "z")
    solver.warmstart.active = true
    solver.warmstart.manual = true
    solver.warmstart.scaled = false
    solver.warmstart.repair = true
    return solver
end

function clear_warmstart!(solver::Solver)
    solver.warmstart.active = false
    solver.warmstart.manual = false
    solver.warmstart.scaled = false
    solver.warmstart.repair = false
    return solver
end

function _automatic_warmstart_is_valid!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    mul_upper_symmetric!(work.xbuff, data.P, work.x)
    add_scaled!(work.xbuff, -solver.settings.kkt_static_reg, work.x)
    add_scaled!(work.xbuff, one(T), data.c)
    copyto!(work.kktres, 1, work.xbuff, 1, data.n)
    if data.p > 0
        mul!(work.xbuff, data.At, work.y)
        add_scaled_to!(work.kktres, 1, one(T), work.xbuff, data.n)
        mul!(work.ybuff, data.A, work.x)
        add_scaled!(work.ybuff, -one(T), data.b)
    end
    if data.m > 0
        mul!(work.xbuff, data.Gt, work.z)
        add_scaled_to!(work.kktres, 1, one(T), work.xbuff, data.n)
        mul!(work.ubuff1, data.G, work.x)
        add_scaled!(work.ubuff1, -one(T), data.h)
        add_scaled!(work.ubuff1, one(T), work.s)
    end
    residual = zero(T)
    @inbounds for i in 1:data.n
        residual = max(residual, abs(work.kktres[i]))
    end
    data.p > 0 && (residual = max(residual, inf_norm(work.ybuff)))
    data.m > 0 && (residual = max(residual, inf_norm(work.ubuff1)))
    data_scale = max(
        one(T),
        max(
            inf_norm(data.c),
            max(
                data.p > 0 ? inf_norm(data.b) : zero(T),
                data.m > 0 ? inf_norm(data.h) : zero(T),
            ),
        ),
    )
    return isfinite(residual) && residual <= T(100) * data_scale
end

function _apply_warmstart!(solver::Solver{T}) where {T<:AbstractFloat}
    ws = solver.warmstart
    ws.active || return false
    has_nan(ws.x) && return false
    has_nan(ws.s) && return false
    has_nan(ws.y) && return false
    has_nan(ws.z) && return false

    scaling = solver.scaling
    work = solver.work
    data = solver.data

    if ws.scaled
        copyto!(work.x, ws.x)
        copyto!(work.s, ws.s)
        copyto!(work.y, ws.y)
        copyto!(work.z, ws.z)
    else
        @inbounds for i in eachindex(work.x, ws.x, scaling.Druiz)
            work.x[i] = safe_div(ws.x[i], scaling.Druiz[i])
        end
        @inbounds for i in eachindex(work.y, ws.y, scaling.Eruiz)
            work.y[i] = safe_div(ws.y[i] * scaling.k, scaling.Eruiz[i])
        end
    end
    if ws.manual && !ws.scaled
        @inbounds for i in eachindex(work.s, ws.s, scaling.Fruiz)
            work.s[i] = ws.s[i] * scaling.Fruiz[i]
        end
        @inbounds for i in eachindex(work.z, ws.z, scaling.Fruiz)
            work.z[i] = safe_div(ws.z[i] * scaling.k, scaling.Fruiz[i])
        end
    elseif ws.scaled && !ws.repair
        # With unchanged data, retain the previous scaled iterate exactly.
    elseif data.m > 0
        mul!(work.ubuff1, data.G, work.x)
        copyto!(work.s, data.h)
        add_scaled!(work.s, -one(T), work.ubuff1)
        if solver.settings.warm_start_mode == :primal
            fill!(work.y, zero(T))
            fill!(work.z, zero(T))
        end
    else
        fill!(work.s, zero(T))
        fill!(work.z, zero(T))
    end
    if ws.manual
        bring2cone!(work.s, data.l, data.q)
        bring2cone!(work.z, data.l, data.q)
    elseif ws.repair
        margin = max(T(1e-4), sqrt(eps(T)))
        bring2cone_strict!(work.s, data.l, data.q, margin)
        bring2cone_strict!(work.z, data.l, data.q, margin)
    end
    if !ws.manual && ws.repair &&
       solver.settings.warm_start_mode in (:primal_dual, :adaptive) &&
       !_automatic_warmstart_is_valid!(solver)
        ws.active = false
        return false
    end
    work.a = one(T)
    return true
end

function _cache_solution_as_warmstart!(solver::Solver{T}) where {T<:AbstractFloat}
    solver.solution.status == QOCO_NUMERICAL_ERROR && return solver
    has_nan(solver.solution.x) && return solver
    has_nan(solver.solution.s) && return solver
    has_nan(solver.solution.y) && return solver
    has_nan(solver.solution.z) && return solver
    mode = solver.settings.warm_start_mode
    if mode == :none
        solver.warmstart.active = false
        solver.warmstart.manual = false
        solver.warmstart.scaled = false
        solver.warmstart.repair = false
        return solver
    end
    copyto!(solver.warmstart.x, solver.work.x)
    copyto!(solver.warmstart.s, solver.work.s)
    copyto!(solver.warmstart.y, solver.work.y)
    copyto!(solver.warmstart.z, solver.work.z)
    solver.warmstart.active = true
    solver.warmstart.manual = false
    solver.warmstart.scaled = true
    solver.warmstart.repair = false
    return solver
end

function update_vector_data!(
    solver::Solver{T};
    c::Union{Nothing,AbstractVector{T}} = nothing,
    b::Union{Nothing,AbstractVector{T}} = nothing,
    h::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    data = solver.data
    scaling = solver.scaling
    c === nothing || length(c) == data.n || throw(ArgumentError("length(c) must match the solver variable dimension"))
    b === nothing || length(b) == data.p || throw(ArgumentError("length(b) must match the solver equality dimension"))
    h === nothing || length(h) == data.m || throw(ArgumentError("length(h) must match the solver cone dimension"))

    solver.solution.status = QOCO_UNSOLVED
    solver.solution.status_detail = ""
    solver.warmstart.repair = true
    if c !== nothing
        copyto!(data.c, c)
        scale!(data.c, scaling.k)
        ew_product!(data.c, data.c, scaling.Druiz)
    end
    if b !== nothing
        copyto!(data.b, b)
        ew_product!(data.b, data.b, scaling.Eruiz)
    end
    if h !== nothing
        copyto!(data.h, h)
        ew_product!(data.h, data.h, scaling.Fruiz)
    end
    data.stats_dirty = true
    return solver
end

function _unscale_problem_data!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    scaling = solver.scaling

    unregularize_P!(data.P, solver.settings.kkt_static_reg)
    if nnz(data.P) > 0
        scale!(data.P.nzval, scaling.kinv)
        row_col_scale_matrix!(data.P, scaling.Dinvruiz, scaling.Dinvruiz)
    end
    scale!(data.c, scaling.kinv)
    ew_product!(data.c, data.c, scaling.Dinvruiz)

    if nnz(data.A) > 0
        row_col_scale_matrix!(data.A, scaling.Einvruiz, scaling.Dinvruiz)
        row_col_scale_matrix!(data.At, scaling.Dinvruiz, scaling.Einvruiz)
    end
    if nnz(data.G) > 0
        row_col_scale_matrix!(data.G, scaling.Finvruiz, scaling.Dinvruiz)
        row_col_scale_matrix!(data.Gt, scaling.Dinvruiz, scaling.Finvruiz)
    end
    if !isempty(data.b)
        ew_product!(data.b, data.b, scaling.Einvruiz)
    end
    if !isempty(data.h)
        ew_product!(data.h, data.h, scaling.Finvruiz)
    end
    return solver
end

function update_matrix_data!(
    solver::Solver{T};
    Px::Union{Nothing,AbstractVector{T}} = nothing,
    Ax::Union{Nothing,AbstractVector{T}} = nothing,
    Gx::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    data = solver.data
    Px === nothing || length(Px) == nnz(data.P) - length(data.Padded_idx) || throw(ArgumentError("Px must match the original nnz(P) before regularization"))
    Ax === nothing || length(Ax) == nnz(data.A) || throw(ArgumentError("Ax must match nnz(A)"))
    Gx === nothing || length(Gx) == nnz(data.G) || throw(ArgumentError("Gx must match nnz(G)"))

    solver.solution.status = QOCO_UNSOLVED
    solver.solution.status_detail = ""
    solver.warmstart.repair = true
    if solver.settings.scaling_mode != :recompute
        if Px !== nothing
            _copy_scaled_P_values!(solver, Px)
        end
        if Ax !== nothing
            _copy_scaled_A_values!(solver, Ax)
        end
        if Gx !== nothing
            _copy_scaled_G_values!(solver, Gx)
        end
        data.stats_dirty = true
        _refresh_static_kkt!(solver)
        return solver
    end

    if solver.warmstart.active && solver.warmstart.scaled
        copyto!(solver.warmstart.x, solver.solution.x)
        copyto!(solver.warmstart.s, solver.solution.s)
        copyto!(solver.warmstart.y, solver.solution.y)
        copyto!(solver.warmstart.z, solver.solution.z)
        solver.warmstart.scaled = false
    end
    _unscale_problem_data!(solver)

    if Px !== nothing
        copy_original_P_values!(data.P, Px, data.Padded_idx)
    end
    if Ax !== nothing
        copyto!(data.A.nzval, Ax)
        @inbounds for i in eachindex(data.At.nzval, data.AtoAt)
            data.At.nzval[i] = Ax[data.AtoAt[i]]
        end
    end
    if Gx !== nothing
        copyto!(data.G.nzval, Gx)
        @inbounds for i in eachindex(data.Gt.nzval, data.GtoGt)
            data.Gt.nzval[i] = Gx[data.GtoGt[i]]
        end
    end

    ruiz_equilibration!(data, solver.scaling, solver.settings.ruiz_iters)
    regularize_existing_P!(data.P, solver.settings.kkt_static_reg)
    _refresh_static_kkt!(solver)
    return solver
end

function _copy_scaled_P_values!(
    solver::Solver{T},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    data = solver.data
    scaling = solver.scaling
    padded_position = 1
    next_padded = isempty(data.Padded_idx) ? typemax(eltype(data.Padded_idx)) :
                  data.Padded_idx[padded_position]
    source = 1
    @inbounds for position in eachindex(data.P.nzval)
        row = data.P.rowval[position]
        col = data.Pcol[position]
        if position == next_padded
            value = zero(T)
            padded_position += 1
            next_padded = padded_position <= length(data.Padded_idx) ?
                          data.Padded_idx[padded_position] : typemax(eltype(data.Padded_idx))
        else
            value = values[source]
            source += 1
        end
        data.P.nzval[position] =
            value * scaling.k * scaling.Druiz[row] * scaling.Druiz[col] +
            ifelse(row == col, solver.settings.kkt_static_reg, zero(T))
    end
    return data.P
end

function _copy_scaled_A_values!(
    solver::Solver{T},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    data = solver.data
    scaling = solver.scaling
    @inbounds for position in eachindex(data.A.nzval, values)
        row = data.A.rowval[position]
        col = data.Acol[position]
        data.A.nzval[position] =
            values[position] * scaling.Eruiz[row] * scaling.Druiz[col]
    end
    @inbounds for position in eachindex(data.At.nzval, data.AtoAt)
        data.At.nzval[position] = data.A.nzval[data.AtoAt[position]]
    end
    return data.A
end

function _copy_scaled_G_values!(
    solver::Solver{T},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    data = solver.data
    scaling = solver.scaling
    @inbounds for position in eachindex(data.G.nzval, values)
        row = data.G.rowval[position]
        col = data.Gcol[position]
        data.G.nzval[position] =
            values[position] * scaling.Fruiz[row] * scaling.Druiz[col]
    end
    @inbounds for position in eachindex(data.Gt.nzval, data.GtoGt)
        data.Gt.nzval[position] = data.G.nzval[data.GtoGt[position]]
    end
    return data.G
end

@inline function _invalidate_solution!(solver::Solver)
    solver.solution.status = QOCO_UNSOLVED
    solver.solution.status_detail = ""
    solver.data.stats_dirty = true
    solver.warmstart.repair = true
    return nothing
end

@inline function _update_static_value!(
    solver::Solver{T,Ti},
    static_position::Integer,
    value::T,
) where {T<:AbstractFloat,Ti<:Integer}
    solver.linsys.factor === nothing && return nothing
    solver.linsys.static_values[static_position] = value
    factor_position = solver.linsys.static2kkt[static_position]
    QDLDL.update_value_internal!(solver.linsys.factor, factor_position, value)
    return nothing
end

function update_c_entries!(
    solver::Solver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    length(indices) == length(values) || throw(DimensionMismatch("indices and values must have equal length"))
    _invalidate_solution!(solver)
    @inbounds for k in eachindex(indices, values)
        index = indices[k]
        solver.data.c[index] =
            values[k] * solver.scaling.k * solver.scaling.Druiz[index]
    end
    return solver
end

function update_b_entries!(
    solver::Solver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    length(indices) == length(values) || throw(DimensionMismatch("indices and values must have equal length"))
    _invalidate_solution!(solver)
    @inbounds for k in eachindex(indices, values)
        index = indices[k]
        solver.data.b[index] = values[k] * solver.scaling.Eruiz[index]
    end
    return solver
end

function update_h_entries!(
    solver::Solver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    length(indices) == length(values) || throw(DimensionMismatch("indices and values must have equal length"))
    _invalidate_solution!(solver)
    @inbounds for k in eachindex(indices, values)
        index = indices[k]
        solver.data.h[index] = values[k] * solver.scaling.Fruiz[index]
    end
    return solver
end

function update_P_entries!(
    solver::Solver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    length(indices) == length(values) || throw(DimensionMismatch("indices and values must have equal length"))
    solver.settings.scaling_mode == :recompute &&
        throw(ArgumentError("indexed matrix updates require scaling_mode=:none or :once"))
    _invalidate_solution!(solver)
    @inbounds for k in eachindex(indices, values)
        position = indices[k]
        row = solver.data.P.rowval[position]
        col = solver.data.Pcol[position]
        scaled_value =
            values[k] * solver.scaling.k *
            solver.scaling.Druiz[row] * solver.scaling.Druiz[col] +
            ifelse(row == col, solver.settings.kkt_static_reg, zero(T))
        solver.data.P.nzval[position] = scaled_value
        _update_static_value!(solver, position, scaled_value)
    end
    return solver
end

function update_A_entries!(
    solver::Solver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    length(indices) == length(values) || throw(DimensionMismatch("indices and values must have equal length"))
    solver.settings.scaling_mode == :recompute &&
        throw(ArgumentError("indexed matrix updates require scaling_mode=:none or :once"))
    _invalidate_solution!(solver)
    p_offset = nnz(solver.data.P)
    @inbounds for k in eachindex(indices, values)
        position = indices[k]
        row = solver.data.A.rowval[position]
        col = solver.data.Acol[position]
        scaled_value =
            values[k] * solver.scaling.Eruiz[row] * solver.scaling.Druiz[col]
        solver.data.A.nzval[position] = scaled_value
        transpose_position = solver.data.AfromAt[position]
        solver.data.At.nzval[transpose_position] = scaled_value
        _update_static_value!(solver, p_offset + transpose_position, scaled_value)
    end
    return solver
end

function update_G_entries!(
    solver::Solver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    length(indices) == length(values) || throw(DimensionMismatch("indices and values must have equal length"))
    solver.settings.scaling_mode == :recompute &&
        throw(ArgumentError("indexed matrix updates require scaling_mode=:none or :once"))
    _invalidate_solution!(solver)
    g_offset = nnz(solver.data.P) + nnz(solver.data.At)
    @inbounds for k in eachindex(indices, values)
        position = indices[k]
        row = solver.data.G.rowval[position]
        col = solver.data.Gcol[position]
        scaled_value =
            values[k] * solver.scaling.Fruiz[row] * solver.scaling.Druiz[col]
        solver.data.G.nzval[position] = scaled_value
        transpose_position = solver.data.GfromGt[position]
        solver.data.Gt.nzval[transpose_position] = scaled_value
        _update_static_value!(solver, g_offset + transpose_position, scaled_value)
    end
    return solver
end
