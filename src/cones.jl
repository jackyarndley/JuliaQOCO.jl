@inline function soc_residual(u::AbstractVector{T}, first::Int, q::Int) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for i in (first + 1):(first + q - 1)
        acc += u[i] * u[i]
    end
    return sqrt(acc) - u[first]
end

@inline function soc_residual2(u::AbstractVector{T}, first::Int, q::Int) where {T<:AbstractFloat}
    acc = u[first] * u[first]
    @inbounds for i in (first + 1):(first + q - 1)
        acc -= u[i] * u[i]
    end
    return acc
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

function set_Wfull_identity!(work::Workspace{T}, data::ProblemData{T}) where {T<:AbstractFloat}
    fill!(work.Wfull, zero(T))
    @inbounds for i in 1:data.l
        work.Wfull[i] = one(T)
    end
    for (block, q) in enumerate(data.q)
        offset = work.Wfull_offsets[block]
        @inbounds for j in 0:(q - 1)
            work.Wfull[offset + j * q + j] = one(T)
        end
    end
    return work.Wfull
end

function nt_multiply!(z::AbstractVector{T}, Wfull::AbstractVector{T}, x::AbstractVector{T}, data::ProblemData{T}, work::Workspace{T,Ti}) where {T<:AbstractFloat,Ti<:Integer}
    @inbounds for i in 1:data.l
        z[i] = Wfull[i] * x[i]
    end
    for (block, q) in enumerate(data.q)
        idx = work.soc_offsets[block]
        offset = work.Wfull_offsets[block]
        @inbounds for j in 0:(q - 1)
            acc = zero(T)
            coloff = offset + j * q
            for k in 0:(q - 1)
                acc += Wfull[coloff + k] * x[idx + k]
            end
            z[idx + j] = acc
        end
    end
    return z
end

function nt_multiply_from!(z::AbstractVector{T}, Wfull::AbstractVector{T}, x::AbstractVector{T}, xoffset::Int, data::ProblemData{T}, work::Workspace{T,Ti}) where {T<:AbstractFloat,Ti<:Integer}
    @inbounds for i in 1:data.l
        z[i] = Wfull[i] * x[xoffset + i - 1]
    end
    for (block, q) in enumerate(data.q)
        idx = work.soc_offsets[block]
        offset = work.Wfull_offsets[block]
        @inbounds for j in 0:(q - 1)
            acc = zero(T)
            coloff = offset + j * q
            for k in 0:(q - 1)
                acc += Wfull[coloff + k] * x[xoffset + idx + k - 1]
            end
            z[idx + j] = acc
        end
    end
    return z
end

function compute_nt_scaling!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    @inbounds for i in 1:data.l
        w2 = safe_div(work.s[i], work.z[i])
        work.WtW[i] = w2
        w = sqrt(w2)
        work.Wfull[i] = w
        work.Winvfull[i] = safe_div(one(T), w)
    end

    for (block, q) in enumerate(data.q)
        idx = work.soc_offsets[block]
        foffset = work.Wfull_offsets[block]
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
        invscale = safe_div(one(T), scale)
        scale2 = scale * scale
        shift = 0
        @inbounds for j in 1:q
            for k in 1:j
                wval = T(2) * work.zbar[k] * work.zbar[j]
                winvval = (j > 1 && k == 1) ? -wval : wval
                if j == 1 && k == 1
                    wval -= one(T)
                    winvval -= one(T)
                elseif j == k
                    wval += one(T)
                    winvval += one(T)
                end
                wval *= scale
                winvval *= invscale
                pos1 = foffset + (j - 1) * q + (k - 1)
                pos2 = foffset + (k - 1) * q + (j - 1)
                work.Wfull[pos1] = wval
                work.Wfull[pos2] = wval
                work.Winvfull[pos1] = winvval
                work.Winvfull[pos2] = winvval
                shift += 1
            end
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

    nt_multiply!(work.lambda, work.Wfull, work.z, data, work)
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

function exact_linesearch(u::AbstractVector{T}, Du::AbstractVector{T}, l::Int, f::T) where {T<:AbstractFloat}
    minval = zero(T)
    @inbounds for i in 1:l
        if Du[i] < minval * u[i]
            minval = Du[i] / u[i]
        end
    end
    return -f < minval ? f : -safe_div(f, minval)
end

function exact_linesearch_from(u::AbstractVector{T}, Du::AbstractVector{T}, Du_offset::Int, l::Int, f::T) where {T<:AbstractFloat}
    minval = zero(T)
    @inbounds for i in 1:l
        dui = Du[Du_offset + i - 1]
        if dui < minval * u[i]
            minval = dui / u[i]
        end
    end
    return -f < minval ? f : -safe_div(f, minval)
end

function bisection_search!(solver::Solver{T}, u::AbstractVector{T}, Du::AbstractVector{T}, f::T) where {T<:AbstractFloat}
    work = solver.work
    data = solver.data
    axpy_to!(work.ubuff1, safe_div(one(T), f), Du, u)
    if cone_residual(work.ubuff1, data.l, data.q) < zero(T)
        return one(T)
    end
    al = zero(T)
    au = one(T)
    a = zero(T)
    for _ in 1:solver.settings.bisect_iters
        a = T(0.5) * (al + au)
        axpy_to!(work.ubuff1, safe_div(a, f), Du, u)
        if cone_residual(work.ubuff1, data.l, data.q) >= zero(T)
            au = a
        else
            al = a
        end
    end
    return al
end

function bisection_search_from!(solver::Solver{T}, u::AbstractVector{T}, Du::AbstractVector{T}, Du_offset::Int, f::T) where {T<:AbstractFloat}
    work = solver.work
    data = solver.data
    axpy_to_from!(work.ubuff1, safe_div(one(T), f), Du, Du_offset, u)
    if cone_residual(work.ubuff1, data.l, data.q) < zero(T)
        return one(T)
    end
    al = zero(T)
    au = one(T)
    a = zero(T)
    for _ in 1:solver.settings.bisect_iters
        a = T(0.5) * (al + au)
        axpy_to_from!(work.ubuff1, safe_div(a, f), Du, Du_offset, u)
        if cone_residual(work.ubuff1, data.l, data.q) >= zero(T)
            au = a
        else
            al = a
        end
    end
    return al
end

function linesearch!(solver::Solver{T}, u::AbstractVector{T}, Du::AbstractVector{T}, f::T) where {T<:AbstractFloat}
    return isempty(solver.data.q) ? exact_linesearch(u, Du, solver.data.l, f) : bisection_search!(solver, u, Du, f)
end

function linesearch_from!(solver::Solver{T}, u::AbstractVector{T}, Du::AbstractVector{T}, Du_offset::Int, f::T) where {T<:AbstractFloat}
    return isempty(solver.data.q) ? exact_linesearch_from(u, Du, Du_offset, solver.data.l, f) : bisection_search_from!(solver, u, Du, Du_offset, f)
end
