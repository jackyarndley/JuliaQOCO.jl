function _refresh_static_kkt!(solver::CoreSolver{T}) where {T<:AbstractFloat}
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
    solver::CoreSolver{T};
    x = nothing,
    s = nothing,
    y = nothing,
    z = nothing,
) where {T<:AbstractFloat}
    for (name, value) in (("x", x), ("s", s), ("y", y), ("z", z))
        value === nothing && continue
        all(isfinite, value) ||
            throw(ArgumentError("Warm-start $name must contain only finite values"))
    end
    usable = solver.solution.result_available
    fallback_x = usable ? solver.solution.x : T[]
    fallback_s = usable ? solver.solution.s : T[]
    fallback_y = usable ? solver.solution.y : T[]
    fallback_z = usable ? solver.solution.z : T[]
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

function clear_warmstart!(solver::CoreSolver)
    solver.warmstart.active = false
    solver.warmstart.manual = false
    solver.warmstart.scaled = false
    solver.warmstart.repair = false
    return solver
end

# Largest relative residual, in original units, of the point currently held in
# `work`. It reuses exactly the residual and reference-norm code of the
# stopping criteria, but normalizes by the data scale rather than by the
# requested tolerance: a warm start is not supposed to be an optimum, so the
# question is whether it is on the right scale, not whether it is converged.
# Returns `Inf` for a point that is not finite or not in the cone.
function _warmstart_relative_residual(solver::CoreSolver{T}) where {T<:AbstractFloat}
    saved_a = solver.work.a
    compute_kkt_residual!(solver)
    assess_iterate!(solver)
    solver.work.a = saved_a
    sol = solver.solution
    sol.cone_valid || return T(Inf)
    (isfinite(sol.pres) && isfinite(sol.dres) && isfinite(sol.gap)) || return T(Inf)
    p_ref, d_ref, g_ref = _stopping_references(solver)
    return max(
        sol.pres / max(one(T), p_ref),
        sol.dres / max(one(T), d_ref),
        abs(sol.gap) / g_ref,
    )
end

# Centrality of the reused point: the smallest per-block complementarity
# relative to the average. A value near one is a well-centered point that the
# interior-point method can improve on immediately; a value near zero means
# some block is already jammed against its boundary, which usually costs more
# iterations than starting cold.
function _cone_centrality(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    m = Int(data.m)
    m == 0 && return one(T)
    total = dot(work.s, work.z)
    total > zero(T) || return zero(T)
    mu = total / T(m)
    worst = T(Inf)
    @inbounds for i in 1:Int(data.l)
        worst = min(worst, work.s[i] * work.z[i])
    end
    for (block, qk) in enumerate(data.q)
        idx = work.soc_offsets[block]
        acc = zero(T)
        @inbounds for k in 0:(qk - 1)
            acc += work.s[idx + k] * work.z[idx + k]
        end
        worst = min(worst, acc / T(qk))
    end
    isfinite(worst) || return one(T)
    return max(worst, zero(T)) / mu
end

# The single coordinate conversion from original units into the internal
# scaled coordinates, for all four components together:
#
#     x_hat = D^-1 x,  s_hat = F s,  y_hat = k E^-1 y,  z_hat = k F^-1 z
#
# It is the exact inverse of `unscaled_solution!`. Splitting it across
# branches is what previously allowed z to be left behind in an old scaling.
function _warmstart_to_scaled!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    ws = solver.warmstart
    scaling = solver.scaling
    work = solver.work
    @inbounds for i in eachindex(work.x, ws.x, scaling.Dinvruiz)
        work.x[i] = ws.x[i] * scaling.Dinvruiz[i]
    end
    @inbounds for i in eachindex(work.s, ws.s, scaling.Fruiz)
        work.s[i] = ws.s[i] * scaling.Fruiz[i]
    end
    @inbounds for i in eachindex(work.y, ws.y, scaling.Einvruiz)
        work.y[i] = ws.y[i] * scaling.Einvruiz[i] * scaling.k
    end
    @inbounds for i in eachindex(work.z, ws.z, scaling.Finvruiz)
        work.z[i] = ws.z[i] * scaling.Finvruiz[i] * scaling.k
    end
    return nothing
end

# Inverse of the above, applied to a cached start still held in the scaled
# coordinates of a scaling that is about to be discarded.
function _warmstart_to_original!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    ws = solver.warmstart
    scaling = solver.scaling
    @inbounds for i in eachindex(ws.x, scaling.Druiz)
        ws.x[i] *= scaling.Druiz[i]
    end
    @inbounds for i in eachindex(ws.s, scaling.Finvruiz)
        ws.s[i] *= scaling.Finvruiz[i]
    end
    @inbounds for i in eachindex(ws.y, scaling.Eruiz)
        ws.y[i] *= scaling.Eruiz[i] * scaling.kinv
    end
    @inbounds for i in eachindex(ws.z, scaling.Fruiz)
        ws.z[i] *= scaling.Fruiz[i] * scaling.kinv
    end
    ws.scaled = false
    return nothing
end

# Cold, strictly interior dual pair for the primal-only reuse modes: y = 0 and
# z = e, the identity element of the cone.
function _reset_duals_to_interior!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    fill!(work.y, zero(T))
    fill!(work.z, zero(T))
    @inbounds for i in 1:Int(data.l)
        work.z[i] = one(T)
    end
    for block in eachindex(data.q)
        work.z[work.soc_offsets[block]] = one(T)
    end
    return nothing
end

# Reconstruct the slack from the primal point. This is a policy choice kept
# deliberately separate from the coordinate conversion above: it produces a
# slack consistent with the current x, which is normally a better start than a
# slack carried over from an older linearization.
function _reconstruct_slack!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    data.m > 0 || return nothing
    mul!(work.ubuff1, data.G, work.x)
    copyto!(work.s, data.h)
    add_scaled!(work.s, -one(T), work.ubuff1)
    return nothing
end

# Relative residual, measured against the data scale in original units, above
# which a reused start is discarded. A start this far off is no longer telling
# the solver anything useful about the new problem.
const WARMSTART_RESIDUAL_LIMIT = 1e2
# Minimum centrality accepted by :adaptive. Below it the reused duals are
# dropped in favour of a fresh, well-centered pair.
const ADAPTIVE_CENTRALITY_LIMIT = 1e-6

function _apply_warmstart!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    ws = solver.warmstart
    profile = solver.solution.profile
    ws.active || return false
    mode = solver.settings.warm_start_mode
    # `:none` disables reuse of a previous solution. An explicit manual start
    # is still honoured, since the caller asked for that point directly.
    if mode == :none && !ws.manual
        ws.active = false
        return false
    end
    if has_nonfinite(ws.x) || has_nonfinite(ws.s) ||
       has_nonfinite(ws.y) || has_nonfinite(ws.z)
        ws.active = false
        profile.warmstart_rejected += 1
        return false
    end

    work = solver.work
    data = solver.data

    if ws.scaled
        # The cached start is already in the current scaled coordinates and the
        # data has not changed, so it is reused exactly.
        copyto!(work.x, ws.x)
        copyto!(work.s, ws.s)
        copyto!(work.y, ws.y)
        copyto!(work.z, ws.z)
    else
        _warmstart_to_scaled!(solver)
    end

    reuse_duals = ws.manual || mode in (:primal_dual, :adaptive)
    if !ws.manual && ws.repair
        # After a data change the cached slack belongs to the old constraint
        # data, so rebuild it from x.
        _reconstruct_slack!(solver)
    end
    if !ws.manual && !reuse_duals
        _reset_duals_to_interior!(solver)
    end

    repaired = false
    if ws.manual
        bring2cone!(work.s, data.l, data.q)
        bring2cone!(work.z, data.l, data.q)
        bring2cone_strict!(work.s, data.l, data.q, relative_cone_margin(work.s))
        bring2cone_strict!(work.z, data.l, data.q, relative_cone_margin(work.z))
        repaired = true
    elseif ws.repair
        bring2cone_strict!(work.s, data.l, data.q, relative_cone_margin(work.s))
        bring2cone_strict!(work.z, data.l, data.q, relative_cone_margin(work.z))
        repaired = true
    end

    if !ws.manual && ws.repair
        limit = T(WARMSTART_RESIDUAL_LIMIT)
        residual = _warmstart_relative_residual(solver)
        # :adaptive makes a real, inexpensive choice: if the transformed point
        # is either far off or badly uncentered, the stale duals are dropped
        # and primal-only reuse is tried once before giving up entirely.
        needs_retry = residual > limit ||
                      (mode == :adaptive && reuse_duals &&
                       _cone_centrality(solver) < T(ADAPTIVE_CENTRALITY_LIMIT))
        if needs_retry && reuse_duals && mode == :adaptive
            _reset_duals_to_interior!(solver)
            bring2cone_strict!(work.z, data.l, data.q, relative_cone_margin(work.z))
            profile.warmstart_retries += 1
            residual = _warmstart_relative_residual(solver)
        end
        if !(residual <= limit)
            ws.active = false
            profile.warmstart_rejected += 1
            return false
        end
    end

    profile.warmstart_accepted += 1
    repaired && (profile.warmstart_repaired += 1)
    work.a = one(T)
    return true
end

# Warm-start caching policy: only a result that was actually published, is
# finite and lies in the cone may seed the next solve. Anything else clears
# the cache, so the next solve starts cold rather than from a point the solver
# itself could not certify.
function _cache_solution_as_warmstart!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    if solver.settings.warm_start_mode == :none || !solver.solution.result_available
        return clear_warmstart!(solver)
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

# Numerical updates.
#
# Every entry point below follows the same contract: all validation happens
# before any committed state is touched, so a rejected update leaves the
# solver exactly as it was. The only exception is the convexity test, which
# needs the new Hessian to exist; that path snapshots the previous values and
# restores them if the test fails.

@inline function _require_finite(values, name::AbstractString)
    values === nothing && return nothing
    all(isfinite, values) ||
        throw(ArgumentError("$name must contain only finite values"))
    return nothing
end

@inline function _require_indices(
    indices::AbstractVector{<:Integer},
    values::AbstractVector,
    limit::Integer,
    name::AbstractString,
)
    length(indices) == length(values) ||
        throw(DimensionMismatch("indices and values must have equal length"))
    @inbounds for index in indices
        1 <= index <= limit ||
            throw(ArgumentError("$name index $index is out of range 1:$limit"))
    end
    _require_finite(values, name)
    return nothing
end

@inline function _invalidate_solution!(solver::CoreSolver)
    solver.solution.status = QOCO_UNSOLVED
    solver.solution.status_detail = ""
    solver.solution.result_available = false
    solver.data.stats_dirty = true
    solver.warmstart.repair = true
    return nothing
end

# Verify convexity of the Hessian currently stored in `data.P`, restoring
# `snapshot` and the factor values if the test fails so that a rejected update
# cannot leave the solver holding a nonconvex model.
function _verify_convexity_or_restore!(
    solver::CoreSolver{T},
    snapshot::Union{Nothing,Vector{T}},
) where {T<:AbstractFloat}
    settings = solver.settings
    settings.convexity_check == :none && return nothing
    is_positive_semidefinite(
        solver.data.P,
        settings.kkt_static_reg,
        settings.convexity_dense_limit,
    ) && return nothing
    if snapshot !== nothing
        copyto!(solver.data.P.nzval, snapshot)
        _refresh_static_kkt!(solver)
    end
    throw(ArgumentError("updated quadratic objective matrix must be positive semidefinite"))
end

@inline function _needs_convexity_snapshot(solver::CoreSolver)
    return solver.settings.convexity_check != :none
end

function update_vector_data!(
    solver::CoreSolver{T};
    c::Union{Nothing,AbstractVector{T}} = nothing,
    b::Union{Nothing,AbstractVector{T}} = nothing,
    h::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    data = solver.data
    scaling = solver.scaling
    c === nothing || length(c) == data.n || throw(ArgumentError("length(c) must match the solver variable dimension"))
    b === nothing || length(b) == data.p || throw(ArgumentError("length(b) must match the solver equality dimension"))
    h === nothing || length(h) == data.m || throw(ArgumentError("length(h) must match the solver cone dimension"))
    _require_finite(c, "c")
    _require_finite(b, "b")
    _require_finite(h, "h")

    _invalidate_solution!(solver)
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
    return solver
end

function _unscale_problem_data!(solver::CoreSolver{T}) where {T<:AbstractFloat}
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

"""
    update_data!(solver; Px, Ax, Gx, c, b, h)

Apply one numerical update transaction to a solver whose sparsity pattern is
unchanged. Every supplied field is validated before anything is committed, and
all of them are considered together before the scaling is recomputed, so that
a new objective vector participates in the objective scaling of the same
transaction.

`Px`, `Ax` and `Gx` are the nonzero values of the original matrices in their
stored CSC order; `Px` excludes the diagonal entries that regularization
padded into the pattern.
"""
function update_data!(
    solver::CoreSolver{T};
    Px::Union{Nothing,AbstractVector{T}} = nothing,
    Ax::Union{Nothing,AbstractVector{T}} = nothing,
    Gx::Union{Nothing,AbstractVector{T}} = nothing,
    c::Union{Nothing,AbstractVector{T}} = nothing,
    b::Union{Nothing,AbstractVector{T}} = nothing,
    h::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    data = solver.data
    Px === nothing || length(Px) == nnz(data.P) - length(data.Padded_idx) ||
        throw(ArgumentError("Px must match the original nnz(P) before regularization"))
    Ax === nothing || length(Ax) == nnz(data.A) || throw(ArgumentError("Ax must match nnz(A)"))
    Gx === nothing || length(Gx) == nnz(data.G) || throw(ArgumentError("Gx must match nnz(G)"))
    c === nothing || length(c) == data.n || throw(ArgumentError("length(c) must match the solver variable dimension"))
    b === nothing || length(b) == data.p || throw(ArgumentError("length(b) must match the solver equality dimension"))
    h === nothing || length(h) == data.m || throw(ArgumentError("length(h) must match the solver cone dimension"))
    _require_finite(Px, "Px")
    _require_finite(Ax, "Ax")
    _require_finite(Gx, "Gx")
    _require_finite(c, "c")
    _require_finite(b, "b")
    _require_finite(h, "h")

    recompute = solver.settings.scaling_mode == :recompute
    snapshot = Px !== nothing && _needs_convexity_snapshot(solver) ?
               copy(data.P.nzval) : nothing

    _invalidate_solution!(solver)

    if !recompute
        Px === nothing || _copy_scaled_P_values!(solver, Px)
        Ax === nothing || _copy_scaled_A_values!(solver, Ax)
        Gx === nothing || _copy_scaled_G_values!(solver, Gx)
        if Px !== nothing || Ax !== nothing || Gx !== nothing
            _refresh_static_kkt!(solver)
        end
        Px === nothing || _verify_convexity_or_restore!(solver, snapshot)
        return update_vector_data!(solver; c = c, b = b, h = h)
    end

    # A recomputed scaling invalidates the cached scaled warm start, so convert
    # it back to original units first, using the scaling it was stored under.
    if solver.warmstart.active && solver.warmstart.scaled
        _warmstart_to_original!(solver)
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
    # The new objective, right-hand side and bounds are placed before the Ruiz
    # sweep so that the recomputed scaling reflects the complete new data.
    c === nothing || copyto!(data.c, c)
    b === nothing || copyto!(data.b, b)
    h === nothing || copyto!(data.h, h)

    ruiz_equilibration!(data, solver.scaling, solver.settings.ruiz_iters)
    regularize_existing_P!(data.P, solver.settings.kkt_static_reg)
    _refresh_static_kkt!(solver)
    Px === nothing || _verify_convexity_or_restore!(solver, nothing)
    return solver
end

# Retained name for matrix-only transactions.
function update_matrix_data!(
    solver::CoreSolver{T};
    Px::Union{Nothing,AbstractVector{T}} = nothing,
    Ax::Union{Nothing,AbstractVector{T}} = nothing,
    Gx::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    return update_data!(solver; Px = Px, Ax = Ax, Gx = Gx)
end

function _copy_scaled_P_values!(
    solver::CoreSolver{T},
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
    solver::CoreSolver{T},
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
    solver::CoreSolver{T},
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

@inline function _update_static_value!(
    solver::CoreSolver{T,Ti},
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
    solver::CoreSolver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    _require_indices(indices, values, solver.data.n, "c")
    _invalidate_solution!(solver)
    @inbounds for k in eachindex(indices, values)
        index = indices[k]
        solver.data.c[index] =
            values[k] * solver.scaling.k * solver.scaling.Druiz[index]
    end
    return solver
end

function update_b_entries!(
    solver::CoreSolver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    _require_indices(indices, values, solver.data.p, "b")
    _invalidate_solution!(solver)
    @inbounds for k in eachindex(indices, values)
        index = indices[k]
        solver.data.b[index] = values[k] * solver.scaling.Eruiz[index]
    end
    return solver
end

function update_h_entries!(
    solver::CoreSolver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    _require_indices(indices, values, solver.data.m, "h")
    _invalidate_solution!(solver)
    @inbounds for k in eachindex(indices, values)
        index = indices[k]
        solver.data.h[index] = values[k] * solver.scaling.Fruiz[index]
    end
    return solver
end

function update_P_entries!(
    solver::CoreSolver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    _require_indices(indices, values, nnz(solver.data.P), "P")
    solver.settings.scaling_mode == :recompute &&
        throw(ArgumentError("indexed matrix updates require scaling_mode=:none or :once"))
    snapshot = _needs_convexity_snapshot(solver) ? copy(solver.data.P.nzval) : nothing
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
    _verify_convexity_or_restore!(solver, snapshot)
    return solver
end

function update_A_entries!(
    solver::CoreSolver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    _require_indices(indices, values, nnz(solver.data.A), "A")
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
    solver::CoreSolver{T},
    indices::AbstractVector{<:Integer},
    values::AbstractVector{T},
) where {T<:AbstractFloat}
    _require_indices(indices, values, nnz(solver.data.G), "G")
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
