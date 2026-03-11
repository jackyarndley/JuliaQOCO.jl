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
