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
    return solver
end

function clear_warmstart!(solver::Solver)
    solver.warmstart.active = false
    solver.warmstart.manual = false
    return solver
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

    @inbounds for i in eachindex(work.x, ws.x, scaling.Druiz)
        work.x[i] = safe_div(ws.x[i], scaling.Druiz[i])
    end
    @inbounds for i in eachindex(work.y, ws.y, scaling.Eruiz)
        work.y[i] = safe_div(ws.y[i] * scaling.k, scaling.Eruiz[i])
    end
    if ws.manual
        @inbounds for i in eachindex(work.s, ws.s, scaling.Fruiz)
            work.s[i] = ws.s[i] * scaling.Fruiz[i]
        end
        @inbounds for i in eachindex(work.z, ws.z, scaling.Fruiz)
            work.z[i] = safe_div(ws.z[i] * scaling.k, scaling.Fruiz[i])
        end
    elseif data.m > 0
        mul!(work.ubuff1, data.G, work.x)
        copyto!(work.s, data.h)
        add_scaled!(work.s, -one(T), work.ubuff1)
        fill!(work.z, zero(T))
    else
        fill!(work.s, zero(T))
        fill!(work.z, zero(T))
    end
    bring2cone!(work.s, data.l, data.q)
    bring2cone!(work.z, data.l, data.q)
    work.a = one(T)
    return true
end

function _cache_solution_as_warmstart!(solver::Solver{T}) where {T<:AbstractFloat}
    solver.solution.status == QOCO_NUMERICAL_ERROR && return solver
    has_nan(solver.solution.x) && return solver
    has_nan(solver.solution.s) && return solver
    has_nan(solver.solution.y) && return solver
    has_nan(solver.solution.z) && return solver
    copyto!(solver.warmstart.x, solver.solution.x)
    copyto!(solver.warmstart.s, solver.solution.s)
    copyto!(solver.warmstart.y, solver.solution.y)
    copyto!(solver.warmstart.z, solver.solution.z)
    solver.warmstart.active = true
    solver.warmstart.manual = false
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
