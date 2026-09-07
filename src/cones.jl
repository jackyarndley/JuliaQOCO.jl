# Euclidean norm of the tail of one second-order cone block. The plain sum of
# squares is used on the common path; if it overflows, the block is renormalized
# by its largest entry and the norm is recovered exactly.
@inline function soc_tail_norm(u::AbstractVector{T}, first::Int, q::Int) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for i in (first + 1):(first + q - 1)
        acc += u[i] * u[i]
    end
    isfinite(acc) && return sqrt(acc)
    scale = zero(T)
    @inbounds for i in (first + 1):(first + q - 1)
        scale = max(scale, abs(u[i]))
    end
    scale > zero(T) || return zero(T)
    acc = zero(T)
    @inbounds for i in (first + 1):(first + q - 1)
        ratio = u[i] / scale
        acc += ratio * ratio
    end
    return scale * sqrt(acc)
end

@inline function soc_residual(u::AbstractVector{T}, first::Int, q::Int) where {T<:AbstractFloat}
    return soc_tail_norm(u, first, q) - u[first]
end

# Cone determinant u0^2 - |u_tail|^2. The factored form is used because the
# subtraction of two nearly equal squares loses most of its significant digits
# exactly where it matters, at a point close to the cone boundary.
@inline function soc_residual2(u::AbstractVector{T}, first::Int, q::Int) where {T<:AbstractFloat}
    tail = soc_tail_norm(u, first, q)
    head = u[first]
    return (head - tail) * (head + tail)
end

function cone_product!(p::AbstractVector{T}, u::AbstractVector{T}, v::AbstractVector{T}, l::Int, qdims::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    @inbounds for i in 1:l
        p[i] = u[i] * v[i]
    end
    idx = l + 1
    for q in qdims
        acc = u[idx] * v[idx]
        @inbounds for k in 1:(q - 1)
            uk = u[idx + k]
            vk = v[idx + k]
            acc += uk * vk
            p[idx + k] = u[idx] * vk + v[idx] * uk
        end
        p[idx] = acc
        idx += q
    end
    return p
end

function cone_division!(d::AbstractVector{T}, λ::AbstractVector{T}, v::AbstractVector{T}, l::Int, qdims::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    @inbounds for i in 1:l
        d[i] = safe_div(v[i], λ[i])
    end
    idx = l + 1
    for q in qdims
        f = soc_residual2(λ, idx, q)
        finv = safe_div(one(T), f)
        λ0inv = safe_div(one(T), λ[idx])
        λ1v1 = zero(T)
        @inbounds for k in 1:(q - 1)
            λ1v1 += λ[idx + k] * v[idx + k]
        end
        d[idx] = finv * (λ[idx] * v[idx] - λ1v1)
        @inbounds for k in 1:(q - 1)
            d[idx + k] = finv * (-λ[idx + k] * v[idx] + λ0inv * f * v[idx + k] + λ0inv * λ1v1 * λ[idx + k])
        end
        idx += q
    end
    return d
end

function cone_residual(u::AbstractVector{T}, l::Int, qdims::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    res = -T(1e7)
    @inbounds for i in 1:l
        res = max(res, -u[i])
    end
    idx = l + 1
    for q in qdims
        res = max(res, soc_residual(u, idx, q))
        idx += q
    end
    return res
end

function bring2cone!(u::AbstractVector{T}, l::Int, qdims::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    if cone_residual(u, l, qdims) < zero(T)
        return u
    end
    a = zero(T)
    @inbounds for i in 1:l
        a = max(a, -u[i])
    end
    idx = l + 1
    for q in qdims
        a = max(a, soc_residual(u, idx, q))
        idx += q
    end
    shift = one(T) + max(a, zero(T))
    @inbounds for i in 1:l
        u[i] += shift
    end
    idx = l + 1
    for q in qdims
        u[idx] += shift
        idx += q
    end
    return u
end

function nt_multiply_W!(
    z::AbstractVector{T},
    x::AbstractVector{T},
    data::ProblemData{T},
    work::Workspace{T,Ti},
) where {T<:AbstractFloat,Ti<:Integer}
    @inbounds for i in 1:data.l
        z[i] = sqrt(max(work.WtW[i], zero(T))) * x[i]
    end
    for (block, q) in enumerate(data.q)
        idx = work.soc_offsets[block]
        scale = work.nt_scale[block]
        dot_v_x = zero(T)
        @inbounds for k in 0:(q - 1)
            dot_v_x += work.nt_v[idx + k] * x[idx + k]
        end
        @inbounds begin
            z[idx] = scale * (T(2) * work.nt_v[idx] * dot_v_x - x[idx])
            for k in 1:(q - 1)
                z[idx + k] =
                    scale * (T(2) * work.nt_v[idx + k] * dot_v_x + x[idx + k])
            end
        end
    end
    return z
end

function nt_multiply_W_from!(
    z::AbstractVector{T},
    x::AbstractVector{T},
    xoffset::Int,
    data::ProblemData{T},
    work::Workspace{T,Ti},
) where {T<:AbstractFloat,Ti<:Integer}
    @inbounds for i in 1:data.l
        z[i] = sqrt(max(work.WtW[i], zero(T))) * x[xoffset + i - 1]
    end
    for (block, q) in enumerate(data.q)
        idx = work.soc_offsets[block]
        scale = work.nt_scale[block]
        dot_v_x = zero(T)
        @inbounds for k in 0:(q - 1)
            dot_v_x += work.nt_v[idx + k] * x[xoffset + idx + k - 1]
        end
        @inbounds begin
            z[idx] = scale * (
                T(2) * work.nt_v[idx] * dot_v_x -
                x[xoffset + idx - 1]
            )
            for k in 1:(q - 1)
                z[idx + k] = scale * (
                    T(2) * work.nt_v[idx + k] * dot_v_x +
                    x[xoffset + idx + k - 1]
                )
            end
        end
    end
    return z
end

function nt_multiply_Winv!(
    z::AbstractVector{T},
    x::AbstractVector{T},
    data::ProblemData{T},
    work::Workspace{T,Ti},
) where {T<:AbstractFloat,Ti<:Integer}
    @inbounds for i in 1:data.l
        z[i] = safe_div(one(T), sqrt(max(work.WtW[i], zero(T)))) * x[i]
    end
    for (block, q) in enumerate(data.q)
        idx = work.soc_offsets[block]
        invscale = safe_div(one(T), work.nt_scale[block])
        dot_jv_x = work.nt_v[idx] * x[idx]
        @inbounds for k in 1:(q - 1)
            dot_jv_x -= work.nt_v[idx + k] * x[idx + k]
        end
        @inbounds begin
            z[idx] =
                invscale * (T(2) * work.nt_v[idx] * dot_jv_x - x[idx])
            for k in 1:(q - 1)
                z[idx + k] = invscale * (
                    -T(2) * work.nt_v[idx + k] * dot_jv_x + x[idx + k]
                )
            end
        end
    end
    return z
end

function compute_nt_scaling!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    @inbounds for i in 1:data.l
        w2 = safe_div(work.s[i], work.z[i])
        work.WtW[i] = w2
        w = sqrt(w2)
    end

    for (block, q) in enumerate(data.q)
        idx = work.soc_offsets[block]
        toffset = work.Wtri_offsets[block]

        s_scal = sqrt(max(soc_residual2(work.s, idx, q), zero(T)))
        z_scal = sqrt(max(soc_residual2(work.z, idx, q), zero(T)))
        sf = safe_div(one(T), s_scal)
        zf = safe_div(one(T), z_scal)
        @inbounds for k in 0:(q - 1)
            work.sbar[k + 1] = sf * work.s[idx + k]
            work.zbar[k + 1] = zf * work.z[idx + k]
        end

        dot_sbar_zbar = zero(T)
        @inbounds for k in 1:q
            dot_sbar_zbar += work.sbar[k] * work.zbar[k]
        end
        gamma = sqrt(T(0.5) * (one(T) + dot_sbar_zbar))
        f = safe_div(one(T), T(2) * gamma)
        work.sbar[1] = f * (work.sbar[1] + work.zbar[1])
        @inbounds for k in 2:q
            work.sbar[k] = f * (work.sbar[k] - work.zbar[k])
        end

        f = safe_div(one(T), sqrt(T(2) * (work.sbar[1] + one(T))))
        work.zbar[1] = f * (work.sbar[1] + one(T))
        @inbounds for k in 2:q
            work.zbar[k] = f * work.sbar[k]
        end

        scale = sqrt(safe_div(s_scal, z_scal))
        scale2 = scale * scale
        work.nt_scale[block] = scale
        @inbounds for k in 0:(q - 1)
            work.nt_v[idx + k] = work.zbar[k + 1]
        end

        if work.soc_expanded[block]
            # The dense upper triangle is replaced by q diagonal entries plus
            # the rank-two expansion in `soc_aux`.
            @inbounds for j in 1:q
                work.WtW[toffset + j - 1] = scale2
            end
            soc_expansion_vectors!(work, block, Int(q))
            continue
        end

        gamma_bar = work.zbar[1]
        gamma2 = gamma_bar * gamma_bar
        tail_norm2 = max(gamma2 - one(T), zero(T))
        shift = 0
        @inbounds for j in 1:q
            for k in 1:j
                acc = if j == 1 && k == 1
                    scale2 * (one(T) + T(8) * gamma2 * tail_norm2)
                elseif k == 1
                    zj = work.zbar[j]
                    scale2 * (T(4) * gamma_bar * (T(2) * gamma2 - one(T)) * zj)
                elseif j == k
                    zj = work.zbar[j]
                    scale2 * (one(T) + T(8) * gamma2 * zj * zj)
                else
                    zj = work.zbar[j]
                    zk = work.zbar[k]
                    scale2 * (T(8) * gamma2 * zj * zk)
                end
                work.WtW[toffset + shift] = acc
                shift += 1
            end
        end
    end

    nt_multiply_W!(work.lambda, work.z, data, work)
    return nothing
end

function subtract_e!(x::AbstractVector{T}, a::T, l::Int, qdims::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    @inbounds for i in 1:l
        x[i] -= a
    end
    idx = l + 1
    for q in qdims
        x[idx] -= a
        idx += q
    end
    return x
end

# Fraction-to-boundary rule. Every line search below returns
#
#     min(1, f * alpha_max)
#
# where alpha_max is the exact distance to the cone boundary along the
# direction. Starting from one rather than from f means an unrestricted
# direction, or one whose boundary already lies beyond 1/f, takes the full
# Newton step instead of being damped for no reason.
@inline _fraction_to_boundary(boundary::T, f::T) where {T<:AbstractFloat} =
    min(one(T), f * boundary)

function exact_linesearch(u::AbstractVector{T}, Du::AbstractVector{T}, l::Int, f::T) where {T<:AbstractFloat}
    step = one(T)
    @inbounds for i in 1:l
        direction = Du[i]
        if direction < zero(T)
            step = min(step, _fraction_to_boundary(-u[i] / direction, f))
        end
    end
    return max(step, zero(T))
end

function exact_linesearch_from(u::AbstractVector{T}, Du::AbstractVector{T}, Du_offset::Int, l::Int, f::T) where {T<:AbstractFloat}
    step = one(T)
    @inbounds for i in 1:l
        direction = Du[Du_offset + i - 1]
        if direction < zero(T)
            step = min(step, _fraction_to_boundary(-u[i] / direction, f))
        end
    end
    return max(step, zero(T))
end

# Number of halvings the safeguarded fallback may use to bracket a feasible
# step. Thirty halvings reach 1e-9, far below the resolution of the five-step
# bisection this replaces, which could only ever return zero or a multiple of
# 1/32 and so reported "no step" for any feasible step under 1/32.
const MAX_LINESEARCH_BACKTRACKS = 30

@inline function _trial_point!(
    buffer::AbstractVector{T},
    u::AbstractVector{T},
    Du::AbstractVector{T},
    Du_offset::Int,
    alpha::T,
) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(buffer, u)
        buffer[i] = u[i] + alpha * Du[Du_offset + i - 1]
    end
    return buffer
end

@inline function _is_interior(u::AbstractVector{T}, l::Integer, qdims::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    all_finite(u) || return false
    return cone_residual(u, Int(l), qdims) < zero(T)
end

# Safeguarded fallback used when the analytical solution is not trustworthy.
# It first tries the full step, then backtracks geometrically until it finds a
# strictly feasible point, and finally refines the bracket by bisection. It
# returns zero only when no positive step above the backtracking floor keeps
# the iterate inside the cone, which is a genuine numerical failure rather than
# an artefact of the search resolution.
function safeguarded_linesearch!(
    solver::CoreSolver{T},
    u::AbstractVector{T},
    Du::AbstractVector{T},
    Du_offset::Int,
    f::T,
) where {T<:AbstractFloat}
    work = solver.work
    data = solver.data
    buffer = work.ubuff1

    _trial_point!(buffer, u, Du, Du_offset, one(T))
    _is_interior(buffer, data.l, data.q) && return one(T)

    feasible = zero(T)
    infeasible = one(T)
    alpha = one(T)
    for _ in 1:MAX_LINESEARCH_BACKTRACKS
        alpha *= T(0.5)
        _trial_point!(buffer, u, Du, Du_offset, alpha)
        if _is_interior(buffer, data.l, data.q)
            feasible = alpha
            break
        end
        infeasible = alpha
    end
    feasible > zero(T) || return zero(T)

    for _ in 1:max(solver.settings.bisect_iters, 1)
        alpha = T(0.5) * (feasible + infeasible)
        _trial_point!(buffer, u, Du, Du_offset, alpha)
        if _is_interior(buffer, data.l, data.q)
            feasible = alpha
        else
            infeasible = alpha
        end
    end
    return _fraction_to_boundary(feasible, f)
end

# Fraction of the cone scale by which a repaired point is pushed inside. It has
# to be far above rounding: a point that is merely finitely interior, a few
# ulps off the boundary, is numerically valid but algorithmically stuck, since
# the interior-point method cannot move away from the face it is pinned to.
const CONE_REPAIR_FRACTION = 1e-4

# Push a point strictly inside the cone. The margin is relative to the
# magnitude of the point, so a warm start whose cone entries are of order 1e6
# is not "repaired" by an absolute shift that leaves it on the boundary, and
# one of order 1e-6 is not swamped by it.
function relative_cone_margin(u::AbstractVector{T}) where {T<:AbstractFloat}
    scale = max(one(T), inf_norm(u))
    return max(sqrt(eps(T)), T(CONE_REPAIR_FRACTION) * scale)
end

function bring2cone_strict!(
    u::AbstractVector{T},
    l::Int,
    qdims::AbstractVector{<:Integer},
    margin::T,
) where {T<:AbstractFloat}
    @inbounds for i in 1:l
        u[i] = max(u[i], margin)
    end
    idx = l + 1
    for q in qdims
        u[idx] = max(u[idx], soc_tail_norm(u, idx, q) + margin)
        idx += q
    end
    return u
end

@inline function _smallest_positive_root(a::T, b::T, c::T) where {T<:AbstractFloat}
    # Solve a*α^2 + 2b*α + c = 0. The starting point is strictly
    # interior, so c > 0 and the first positive root is the cone boundary.
    scale = max(abs(a), max(abs(b), abs(c)))
    if scale == zero(T)
        return T(Inf)
    end
    tolerance = T(32) * eps(T) * scale
    if abs(a) <= tolerance
        linear = T(2) * b
        return linear < -tolerance ? -c / linear : T(Inf)
    end
    discriminant = muladd(b, b, -a * c)
    if discriminant < -tolerance * scale
        return T(Inf)
    end
    root_discriminant = sqrt(max(discriminant, zero(T)))
    q = -(b + copysign(root_discriminant, b))
    if q == zero(T)
        root = -b / a
        return root > zero(T) ? root : T(Inf)
    end
    root_1 = q / a
    root_2 = c / q
    root = T(Inf)
    root_1 > zero(T) && (root = root_1)
    root_2 > zero(T) && (root = min(root, root_2))
    return root
end

# Exact distance to the boundary of one second-order cone along u + α*Du.
# The quadratic α^2*(d0^2 - |d|^2) + 2α*(u0*d0 - <u,d>) + (u0^2 - |u|^2) = 0 is
# formed after normalizing both blocks by their largest entry, which keeps all
# three coefficients of order one regardless of the magnitude of the data, and
# the two determinants are evaluated in factored form to avoid the cancellation
# that dominates near the boundary. Returns `nothing` when the starting point
# is not usably interior, so that the caller can fall back to a search.
function _soc_boundary_step(
    u::AbstractVector{T},
    Du::AbstractVector{T},
    Du_offset::Int,
    first::Int,
    qdim::Int,
) where {T<:AbstractFloat}
    u_scale = zero(T)
    d_scale = zero(T)
    @inbounds for k in 0:(qdim - 1)
        u_scale = max(u_scale, abs(u[first + k]))
        d_scale = max(d_scale, abs(Du[Du_offset + first + k - 1]))
    end
    (isfinite(u_scale) && isfinite(d_scale)) || return nothing
    # A zero direction never leaves the cone.
    d_scale > zero(T) || return T(Inf)
    u_scale > zero(T) || return nothing

    u_head = u[first] / u_scale
    d_head = Du[Du_offset + first - 1] / d_scale
    u_tail2 = zero(T)
    d_tail2 = zero(T)
    cross = u_head * d_head
    @inbounds for k in 1:(qdim - 1)
        uk = u[first + k] / u_scale
        dk = Du[Du_offset + first + k - 1] / d_scale
        u_tail2 += uk * uk
        d_tail2 += dk * dk
        cross -= uk * dk
    end
    u_tail = sqrt(u_tail2)
    d_tail = sqrt(d_tail2)
    c = (u_head - u_tail) * (u_head + u_tail)
    a = (d_head - d_tail) * (d_head + d_tail)
    # The root selection relies on the start being strictly interior. Right at
    # the boundary the factored determinant can round to zero or below, and the
    # caller then falls back to a search rather than trusting a root here.
    c > zero(T) || return nothing

    ratio = u_scale / d_scale
    # Substituting β = α / ratio makes the quadratic a*β^2 + 2b*β + c = 0.
    beta = _smallest_positive_root(a, cross, c)
    if !isfinite(beta)
        # A concave quadratic starting positive must cross zero, so a missing
        # root means the coefficients are not trustworthy.
        a < zero(T) && return nothing
        return T(Inf)
    end
    step = beta * ratio
    return isfinite(step) && step >= zero(T) ? step : nothing
end

function _analytical_cone_linesearch(
    u::AbstractVector{T},
    Du::AbstractVector{T},
    Du_offset::Int,
    l::Int,
    qdims::AbstractVector{<:Integer},
    f::T,
) where {T<:AbstractFloat}
    step = one(T)
    @inbounds for i in 1:l
        direction = Du[Du_offset + i - 1]
        if direction < zero(T)
            u[i] > zero(T) || return nothing
            step = min(step, _fraction_to_boundary(-u[i] / direction, f))
        end
    end

    first = l + 1
    for qdim in qdims
        boundary = _soc_boundary_step(u, Du, Du_offset, first, qdim)
        boundary === nothing && return nothing
        isfinite(boundary) && (step = min(step, _fraction_to_boundary(boundary, f)))
        first += qdim
    end
    return isfinite(step) ? clamp(step, zero(T), one(T)) : nothing
end

function linesearch!(solver::CoreSolver{T}, u::AbstractVector{T}, Du::AbstractVector{T}, f::T) where {T<:AbstractFloat}
    isempty(solver.data.q) && return exact_linesearch(u, Du, solver.data.l, f)
    step = _analytical_cone_linesearch(u, Du, 1, solver.data.l, solver.data.q, f)
    return step === nothing ? safeguarded_linesearch!(solver, u, Du, 1, f) : step
end

function linesearch_from!(solver::CoreSolver{T}, u::AbstractVector{T}, Du::AbstractVector{T}, Du_offset::Int, f::T) where {T<:AbstractFloat}
    isempty(solver.data.q) &&
        return exact_linesearch_from(u, Du, Du_offset, solver.data.l, f)
    step = _analytical_cone_linesearch(
        u,
        Du,
        Du_offset,
        solver.data.l,
        solver.data.q,
        f,
    )
    return step === nothing ?
           safeguarded_linesearch!(solver, u, Du, Du_offset, f) : step
end
