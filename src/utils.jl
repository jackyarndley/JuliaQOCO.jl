const SAFE_DIV_EPS = 1e-15

# Denominator below which a ratio is treated as meaningless. The floor is the
# value tuned for `Float64`; the `eps`-proportional term takes over in lower
# precision, where 1e-15 would be far below the noise level and would let a
# ratio built from rounding error through as if it were a real number.
#
# This threshold is only for quantities that have no business being small.
# Anything the barrier drives towards zero - a slack, a dual, a cone
# determinant, a Nesterov-Todd scale - must use `bounded_ratio` instead.
@inline safe_div_eps(::Type{T}) where {T<:AbstractFloat} =
    max(T(SAFE_DIV_EPS), T(4) * eps(T))

@inline elapsed_time_sec(t0::UInt64) = (time_ns() - t0) * 1e-9

# Ratio whose denominator is nonnegative and is *expected* to become small:
# the interior-point method drives slacks, duals and cone determinants towards
# zero, so `s/z` growing without bound is the normal course of a solve, not a
# failure. The quotient is therefore computed whenever it is representable,
# and only a genuine overflow saturates. The sign of the numerator is kept.
#
# Using a fixed epsilon here instead would replace a large but perfectly good
# scaling with a sentinel: in single precision a dual of 1.6e-7 is an ordinary
# iterate, and turning `s/z` into `floatmax` for it destroys the step.
@inline function bounded_ratio(a::T, b::T) where {T<:AbstractFloat}
    iszero(a) && return zero(T)
    (isfinite(a) && isfinite(b) && b > zero(T)) || return copysign(floatmax(T), a)
    value = a / b
    return isfinite(value) ? value : copysign(floatmax(T), a)
end

# Sign-preserving division that reports an unusable denominator as NaN rather
# than fabricating a large value. For denominators that are *not* expected to
# vanish, such as a total complementarity that has already collapsed.
@inline function checked_div(a::T, b::T) where {T<:AbstractFloat}
    (isfinite(a) && isfinite(b) && abs(b) > safe_div_eps(T)) || return T(NaN)
    value = a / b
    return isfinite(value) ? value : T(NaN)
end

# Inverse of a strictly positive scale, clamped to stay finite and positive.
# Used for the Ruiz scale inverses and the objective scale, where a zero or
# nonfinite inverse would silently destroy the unscaling algebra.
@inline function positive_inv(b::T) where {T<:AbstractFloat}
    (isfinite(b) && b > zero(T)) || return one(T)
    value = inv(b)
    return isfinite(value) && value > zero(T) ? value : one(T)
end

function reset_solve_profile!(profile::SolveProfile)
    profile.initialize_time_sec = 0.0
    profile.residual_time_sec = 0.0
    profile.objective_time_sec = 0.0
    profile.mu_time_sec = 0.0
    profile.stopping_time_sec = 0.0
    profile.nt_scaling_time_sec = 0.0
    profile.nt_update_time_sec = 0.0
    profile.predictor_time_sec = 0.0
    profile.linsys_solve_time_sec = 0.0
    profile.linsys_refine_time_sec = 0.0
    profile.linsys_solves = 0
    profile.linsys_refinements = 0
    profile.nt_refactors = 0
    profile.dynamic_regularizations = 0
    profile.regularized_pivots = 0
    profile.factorization_retries = 0
    profile.warmstart_accepted = 0
    profile.warmstart_repaired = 0
    profile.warmstart_rejected = 0
    profile.warmstart_retries = 0
    return profile
end

function copy_negate!(dest::AbstractVector{T}, src::AbstractVector{T}) where {T}
    @inbounds @simd for i in eachindex(dest, src)
        dest[i] = -src[i]
    end
    return dest
end

function ew_product!(dest::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    @inbounds @simd for i in eachindex(dest, x, y)
        dest[i] = x[i] * y[i]
    end
    return dest
end

# Ruiz scale vectors are strictly positive by construction; the clamped
# inverse preserves that invariant even for a degenerate zero entry.
function reciprocal!(dest::AbstractVector{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(dest, x)
        dest[i] = positive_inv(x[i])
    end
    return dest
end

function scale!(x::AbstractVector{T}, α::T) where {T}
    @inbounds @simd for i in eachindex(x)
        x[i] *= α
    end
    return x
end

function scale_to!(dest::AbstractVector{T}, x::AbstractVector{T}, α::T) where {T}
    @inbounds @simd for i in eachindex(dest, x)
        dest[i] = α * x[i]
    end
    return dest
end

function add_scaled!(y::AbstractVector{T}, α::T, x::AbstractVector{T}) where {T}
    @inbounds @simd for i in eachindex(y, x)
        y[i] += α * x[i]
    end
    return y
end

function axpy_to!(dest::AbstractVector{T}, α::T, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    @inbounds @simd for i in eachindex(dest, x, y)
        dest[i] = y[i] + α * x[i]
    end
    return dest
end

function axpy_to_from!(dest::AbstractVector{T}, α::T, x::AbstractVector{T}, xoffset::Int, y::AbstractVector{T}) where {T}
    @inbounds @simd for i in eachindex(dest, y)
        dest[i] = y[i] + α * x[xoffset + i - 1]
    end
    return dest
end

function add_scaled_from!(y::AbstractVector{T}, α::T, x::AbstractVector{T}, xoffset::Int, n::Int) where {T}
    @inbounds @simd for i in 1:n
        y[i] += α * x[xoffset + i - 1]
    end
    return y
end

function add_scaled_to!(y::AbstractVector{T}, yoffset::Int, α::T, x::AbstractVector{T}, n::Int) where {T}
    @inbounds @simd for i in 1:n
        y[yoffset + i - 1] += α * x[i]
    end
    return y
end

function copy_negate_to!(dest::AbstractVector{T}, dest_offset::Int, src::AbstractVector{T}, n::Int) where {T}
    @inbounds @simd for i in 1:n
        dest[dest_offset + i - 1] = -src[i]
    end
    return dest
end

function inf_norm(x::AbstractVector{T}) where {T<:AbstractFloat}
    nrm = zero(T)
    @inbounds for xi in x
        nrm = max(nrm, abs(xi))
    end
    return nrm
end

function weighted_inf_norm(x::AbstractVector{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
    nrm = zero(T)
    @inbounds @simd for i in eachindex(x, w)
        nrm = max(nrm, abs(x[i] * w[i]))
    end
    return nrm
end

function weighted_inf_norm_from(x::AbstractVector{T}, xoffset::Int, w::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    nrm = zero(T)
    @inbounds @simd for i in 1:n
        nrm = max(nrm, abs(x[xoffset + i - 1] * w[i]))
    end
    return nrm
end

# sum_i (x_i w_i) * (y_i w_i): a dot product for two vectors that share the
# same coordinate scaling. The complementarity product between s and z is NOT
# of this form, because those two carry reciprocal row scales; see
# `check_stopping!` for the correct conversion.
function weighted_dot(x::AbstractVector{T}, y::AbstractVector{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds @simd for i in eachindex(x, y, w)
        wi = w[i]
        acc += (x[i] * wi) * (y[i] * wi)
    end
    return acc
end

# Unweighted dot between a slice of a long buffer and a short vector.
function dot_from(x::AbstractVector{T}, xoffset::Int, y::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds @simd for i in 1:n
        acc += x[xoffset + i - 1] * y[i]
    end
    return acc
end

function min_abs_nonzero(x::AbstractVector{T}) where {T<:AbstractFloat}
    mn = floatmax(T)
    @inbounds for xi in x
        ax = abs(xi)
        if ax > zero(T)
            mn = min(mn, ax)
        end
    end
    return mn == floatmax(T) ? zero(T) : mn
end

function has_nan(x::AbstractVector)
    @inbounds for xi in x
        isnan(xi) && return true
    end
    return false
end

# Prefer this over `has_nan` at numerical boundaries: an infinity is as
# unusable as a NaN for warm starts, directions, pivots and final results.
function has_nonfinite(x::AbstractVector)
    @inbounds for xi in x
        isfinite(xi) || return true
    end
    return false
end

all_finite(x::AbstractVector) = !has_nonfinite(x)

function compute_scaling_statistics(data::ProblemData{T}) where {T<:AbstractFloat}
    obj_min = zero(T)
    obj_max = zero(T)
    con_min = zero(T)
    con_max = zero(T)
    rhs_min = zero(T)
    rhs_max = zero(T)

    if nnz(data.P) > 0
        obj_min = min_abs_nonzero(data.P.nzval)
        obj_max = inf_norm(data.P.nzval)
    end
    obj_min = ifelse(obj_min == zero(T), min_abs_nonzero(data.c), min(obj_min, min_abs_nonzero(data.c)))
    obj_max = max(obj_max, inf_norm(data.c))

    if nnz(data.A) > 0
        con_min = min_abs_nonzero(data.A.nzval)
        con_max = inf_norm(data.A.nzval)
    end
    if nnz(data.G) > 0
        gmin = min_abs_nonzero(data.G.nzval)
        gmax = inf_norm(data.G.nzval)
        con_min = con_min == zero(T) ? gmin : min(con_min, gmin)
        con_max = max(con_max, gmax)
    end

    if !isempty(data.b)
        rhs_min = min_abs_nonzero(data.b)
        rhs_max = inf_norm(data.b)
    end
    if !isempty(data.h)
        hmin = min_abs_nonzero(data.h)
        hmax = inf_norm(data.h)
        rhs_min = rhs_min == zero(T) ? hmin : min(rhs_min, hmin)
        rhs_max = max(rhs_max, hmax)
    end

    return ScalingStats{T}(obj_min, obj_max, con_min, con_max, rhs_min, rhs_max)
end
