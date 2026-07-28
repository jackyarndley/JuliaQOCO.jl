function kkt_nt_nnz(l::Integer, qdims::AbstractVector{<:Integer})
    total = l
    for q in qdims
        total += q * (q + 1) ÷ 2
    end
    return total
end

function kkt_nt_full_nnz(l::Integer, qdims::AbstractVector{<:Integer})
    total = l
    for q in qdims
        total += q * q
    end
    return total
end

function construct_kkt(data::ProblemData{T,Ti}, settings::Settings{T}, work::Workspace{T,Ti}) where {T<:AbstractFloat,Ti<:Integer}
    n = Int(data.n)
    p = Int(data.p)
    m = Int(data.m)
    N = n + p + m
    Wnnz = length(work.WtW)
    total_nnz = nnz(data.P) + nnz(data.A) + nnz(data.G) + Wnnz + p

    colptr = Vector{Ti}(undef, N + 1)
    rowval = Vector{Ti}(undef, total_nnz)
    nzval = Vector{T}(undef, total_nnz)
    nt2kkt = Vector{Ti}(undef, Wnnz)
    P2kkt = Vector{Ti}(undef, nnz(data.P))
    At2kkt = Vector{Ti}(undef, nnz(data.At))
    Gt2kkt = Vector{Ti}(undef, nnz(data.Gt))
    ntdiag_positions = Vector{Ti}()
    sizehint!(ntdiag_positions, m)

    colptr[1] = one(Ti)
    nz = 1
    col = 1

    @inbounds for j in 1:n
        for k in data.P.colptr[j]:(data.P.colptr[j + 1] - 1)
            rowval[nz] = data.P.rowval[k]
            nzval[nz] = data.P.nzval[k]
            P2kkt[k] = Ti(nz)
            nz += 1
        end
        colptr[col + 1] = Ti(nz)
        col += 1
    end

    @inbounds for j in 1:p
        for k in data.At.colptr[j]:(data.At.colptr[j + 1] - 1)
            rowval[nz] = data.At.rowval[k]
            nzval[nz] = data.At.nzval[k]
            At2kkt[k] = Ti(nz)
            nz += 1
        end
        rowval[nz] = Ti(n + j)
        nzval[nz] = -settings.kkt_static_reg
        nz += 1
        colptr[col + 1] = Ti(nz)
        col += 1
    end

    ntpos = 1
    @inbounds for j in 1:Int(data.l)
        for k in data.Gt.colptr[j]:(data.Gt.colptr[j + 1] - 1)
            rowval[nz] = data.Gt.rowval[k]
            nzval[nz] = data.Gt.nzval[k]
            Gt2kkt[k] = Ti(nz)
            nz += 1
        end
        rowval[nz] = Ti(n + p + j)
        nzval[nz] = -one(T)
        nt2kkt[ntpos] = Ti(nz)
        push!(ntdiag_positions, Ti(ntpos))
        ntpos += 1
        nz += 1
        colptr[col + 1] = Ti(nz)
        col += 1
    end

    cone_start = Int(data.l) + 1
    for q in data.q
        for global_col in cone_start:(cone_start + q - 1)
            for k in data.Gt.colptr[global_col]:(data.Gt.colptr[global_col + 1] - 1)
                rowval[nz] = data.Gt.rowval[k]
                nzval[nz] = data.Gt.nzval[k]
                Gt2kkt[k] = Ti(nz)
                nz += 1
            end
            local_col = global_col - cone_start + 1
            for local_row in 1:local_col
                rowval[nz] = Ti(n + p + cone_start + local_row - 1)
                nzval[nz] = local_row == local_col ? -one(T) : zero(T)
                nt2kkt[ntpos] = Ti(nz)
                if local_row == local_col
                    push!(ntdiag_positions, Ti(ntpos))
                end
                ntpos += 1
                nz += 1
            end
            colptr[col + 1] = Ti(nz)
            col += 1
        end
        cone_start += q
    end

    K = SparseMatrixCSC(N, N, colptr, rowval, nzval)
    return K, nt2kkt, ntdiag_positions, P2kkt, At2kkt, Gt2kkt
end

function set_identity_scalings!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    fill!(work.WtW, zero(T))
    set_Wfull_identity!(work, data)
    fill!(work.Winvfull, zero(T))
    @inbounds for i in 1:Int(data.l)
        work.WtW[i] = one(T)
        work.Winvfull[i] = one(T)
    end
    for (block, q) in enumerate(data.q)
        toffset = work.Wtri_offsets[block]
        foffset = work.Wfull_offsets[block]
        @inbounds for j in 1:q
            work.WtW[toffset + (j * (j - 1)) ÷ 2 + j - 1] = one(T)
            for k in 1:q
                work.Winvfull[foffset + (j - 1) * q + k - 1] = ifelse(j == k, one(T), zero(T))
            end
        end
        idx = work.soc_offsets[block]
        work.nt_scale[block] = one(T)
        work.nt_v[idx] = one(T)
        @inbounds for k in 1:(q - 1)
            work.nt_v[idx + k] = zero(T)
        end
    end
    return nothing
end

function update_nt_block!(solver::Solver{T}) where {T<:AbstractFloat}
    work = solver.work
    linsys = solver.linsys
    @inbounds for i in eachindex(work.WtW, linsys.nt_values)
        linsys.nt_values[i] = -work.WtW[i]
    end
    @inbounds for pos in linsys.ntdiag_positions
        linsys.nt_values[pos] -= solver.settings.kkt_static_reg
    end
    QDLDL.update_values_internal!(linsys.factor, linsys.nt2kkt, linsys.nt_values)
    QDLDL.refactor!(linsys.factor)
    solver.settings.profile && (solver.solution.profile.nt_refactors += 1)
    return nothing
end

function kkt_multiply!(y::AbstractVector{T}, x::AbstractVector{T}, data::ProblemData{T}, work::Workspace{T}; include_nt::Bool = true) where {T<:AbstractFloat}
    n = Int(data.n)
    p = Int(data.p)
    m = Int(data.m)

    xpr = view(x, 1:n)
    ypr = view(y, 1:n)
    mul_upper_symmetric!(ypr, data.P, xpr)

    if p > 0
        yeq = view(y, (n + 1):(n + p))
        mul!(work.xbuff, data.At, view(x, (n + 1):(n + p)))
        add_scaled!(ypr, one(T), work.xbuff)
        mul!(work.ybuff, data.A, xpr)
        copyto!(yeq, work.ybuff)
    end

    if m > 0
        ycon = view(y, (n + p + 1):(n + p + m))
        mul!(work.xbuff, data.Gt, view(x, (n + p + 1):(n + p + m)))
        add_scaled!(ypr, one(T), work.xbuff)
        mul!(work.ubuff1, data.G, xpr)
        copyto!(ycon, work.ubuff1)
        if include_nt
            nt_multiply_W!(work.ubuff1, view(x, (n + p + 1):(n + p + m)), data, work)
            nt_multiply_W!(work.ubuff2, work.ubuff1, data, work)
            add_scaled!(ycon, -one(T), work.ubuff2)
        end
    end
    return y
end

function _solve_linsys_fast!(
    solver::Solver{T},
    rhs::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:AbstractFloat}
    copyto!(x, rhs)
    QDLDL.solve!(solver.linsys.factor, x)
    solver.settings.iter_ref_iters == 0 && return x
    rhs_norm = max(one(T), inf_norm(rhs))
    for _ in 1:solver.settings.iter_ref_iters
        kkt_multiply!(solver.work.xyzbuff1, x, solver.data, solver.work)
        @inbounds for i in eachindex(x)
            solver.work.xyzbuff1[i] = rhs[i] - solver.work.xyzbuff1[i]
        end
        inf_norm(solver.work.xyzbuff1) <= solver.settings.iter_ref_tol * rhs_norm && break
        QDLDL.solve!(solver.linsys.factor, solver.work.xyzbuff1)
        add_scaled!(x, one(T), solver.work.xyzbuff1)
    end
    return x
end

function _solve_linsys_profiled!(
    solver::Solver{T},
    rhs::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:AbstractFloat}
    profile = solver.solution.profile
    copyto!(x, rhs)
    tsolve = time_ns()
    QDLDL.solve!(solver.linsys.factor, x)
    profile.linsys_solve_time_sec += elapsed_time_sec(tsolve)
    profile.linsys_solves += 1
    solver.settings.iter_ref_iters == 0 && return x
    rhs_norm = max(one(T), inf_norm(rhs))
    for _ in 1:solver.settings.iter_ref_iters
        kkt_multiply!(solver.work.xyzbuff1, x, solver.data, solver.work)
        @inbounds for i in eachindex(x)
            solver.work.xyzbuff1[i] = rhs[i] - solver.work.xyzbuff1[i]
        end
        inf_norm(solver.work.xyzbuff1) <= solver.settings.iter_ref_tol * rhs_norm && break
        trefine = time_ns()
        QDLDL.solve!(solver.linsys.factor, solver.work.xyzbuff1)
        profile.linsys_refine_time_sec += elapsed_time_sec(trefine)
        profile.linsys_refinements += 1
        add_scaled!(x, one(T), solver.work.xyzbuff1)
    end
    return x
end

@inline function solve_linsys!(
    solver::Solver{T},
    rhs::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:AbstractFloat}
    return solver.settings.profile ?
           _solve_linsys_profiled!(solver, rhs, x) :
           _solve_linsys_fast!(solver, rhs, x)
end

function initialize_ipm!(solver::Solver{T}) where {T<:AbstractFloat}
    if _apply_warmstart!(solver)
        return nothing
    end
    data = solver.data
    work = solver.work
    set_identity_scalings!(solver)
    update_nt_block!(solver)
    work.a = one(T)

    copy_negate_to!(work.rhs, 1, data.c, data.n)
    copyto!(work.rhs, data.n + 1, data.b, 1, data.p)
    copyto!(work.rhs, data.n + data.p + 1, data.h, 1, data.m)

    solve_linsys!(solver, work.rhs, work.xyz)
    copyto!(work.x, 1, work.xyz, 1, data.n)
    copyto!(work.y, 1, work.xyz, data.n + 1, data.p)
    copyto!(work.z, 1, work.xyz, data.n + data.p + 1, data.m)
    copy_negate!(work.s, work.z)
    bring2cone!(work.s, data.l, data.q)
    bring2cone!(work.z, data.l, data.q)
    return nothing
end

function compute_kkt_residual!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    scaling = solver.scaling

    mul_upper_symmetric!(work.xbuff, data.P, work.x)
    add_scaled!(work.xbuff, -solver.settings.kkt_static_reg, work.x)
    work.quad_obj = dot(work.xbuff, work.x)
    work.xPx = weighted_dot(work.x, work.xbuff, scaling.Dinvruiz)
    work.Pxinf = weighted_inf_norm(work.xbuff, scaling.Dinvruiz)
    copyto!(work.kktres, 1, work.xbuff, 1, data.n)
    add_scaled_to!(work.kktres, 1, one(T), data.c, data.n)

    if data.p > 0
        mul!(work.xbuff, data.At, work.y)
        work.Atyinf = weighted_inf_norm(work.xbuff, scaling.Dinvruiz)
        add_scaled_to!(work.kktres, 1, one(T), work.xbuff, data.n)
        mul!(work.ybuff, data.A, work.x)
        work.Axinf = weighted_inf_norm(work.ybuff, scaling.Einvruiz)
        copyto!(work.kktres, data.n + 1, work.ybuff, 1, data.p)
        add_scaled_to!(work.kktres, data.n + 1, -one(T), data.b, data.p)
    else
        work.Atyinf = zero(T)
        work.Axinf = zero(T)
    end

    if data.m > 0
        mul!(work.xbuff, data.Gt, work.z)
        work.Gtzinf = weighted_inf_norm(work.xbuff, scaling.Dinvruiz)
        add_scaled_to!(work.kktres, 1, one(T), work.xbuff, data.n)
        mul!(work.ubuff1, data.G, work.x)
        work.Gxinf = weighted_inf_norm(work.ubuff1, scaling.Finvruiz)
        copyto!(work.kktres, data.n + data.p + 1, work.ubuff1, 1, data.m)
        add_scaled_to!(work.kktres, data.n + data.p + 1, -one(T), data.h, data.m)
        add_scaled_to!(work.kktres, data.n + data.p + 1, one(T), work.s, data.m)
    else
        work.Gtzinf = zero(T)
        work.Gxinf = zero(T)
    end
    return nothing
end

function compute_mu!(solver::Solver{T}) where {T<:AbstractFloat}
    solver.work.mu = isempty(solver.work.s) ? zero(T) : safe_div(dot(solver.work.s, solver.work.z), T(length(solver.work.s)))
    return solver.work.mu
end

function compute_objective!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    obj = dot(work.x, data.c) + T(0.5) * work.quad_obj
    solver.solution.obj = safe_div(obj, solver.scaling.k)
    return solver.solution.obj
end

function construct_kkt_aff_rhs!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    copy_negate!(work.rhs, work.kktres)
    nt_multiply_W!(work.ubuff1, work.lambda, data, work)
    add_scaled_to!(work.rhs, data.n + data.p + 1, one(T), work.ubuff1, data.m)
    return nothing
end

function construct_kkt_comb_rhs!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    copy_negate!(work.rhs, work.kktres)
    nt_multiply_Winv!(work.ubuff1, work.Ds, data, work)
    nt_multiply_W_from!(work.ubuff2, work.xyz, data.n + data.p + 1, data, work)
    cone_product!(work.ubuff3, work.ubuff1, work.ubuff2, data.l, data.q)
    subtract_e!(work.ubuff3, work.sigma * work.mu, data.l, data.q)
    cone_product!(work.ubuff1, work.lambda, work.lambda, data.l, data.q)
    copy_negate!(work.Ds, work.ubuff1)
    add_scaled!(work.Ds, -one(T), work.ubuff3)
    cone_division!(work.ubuff2, work.lambda, work.Ds, data.l, data.q)
    nt_multiply_W!(work.ubuff1, work.ubuff2, data, work)
    add_scaled_to!(work.rhs, data.n + data.p + 1, -one(T), work.ubuff1, data.m)
    return nothing
end

function compute_centering!(solver::Solver{T}) where {T<:AbstractFloat}
    if solver.data.m == 0
        solver.work.sigma = zero(T)
        return zero(T)
    end
    data = solver.data
    work = solver.work
    Dz_offset = data.n + data.p + 1
    a = min(linesearch_from!(solver, work.z, work.xyz, Dz_offset, one(T)), linesearch!(solver, work.s, work.Ds, one(T)))
    axpy_to_from!(work.ubuff1, a, work.xyz, Dz_offset, work.z)
    axpy_to!(work.ubuff2, a, work.Ds, work.s)
    rho = safe_div(dot(work.ubuff1, work.ubuff2), dot(work.z, work.s))
    sigma = clamp(rho, zero(T), one(T))
    work.sigma = sigma * sigma * sigma
    return work.sigma
end

function predictor_corrector!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    Dz_offset = data.n + data.p + 1
    construct_kkt_aff_rhs!(solver)
    solve_linsys!(solver, work.rhs, work.xyz)

    nt_multiply_W_from!(work.ubuff1, work.xyz, Dz_offset, data, work)
    @inbounds for i in eachindex(work.ubuff1)
        work.ubuff1[i] = -work.ubuff1[i] - work.lambda[i]
    end
    nt_multiply_W!(work.Ds, work.ubuff1, data, work)

    compute_centering!(solver)
    construct_kkt_comb_rhs!(solver)
    solve_linsys!(solver, work.rhs, work.xyz)

    if has_nan(work.xyz)
        work.a = zero(T)
        return nothing
    end

    cone_division!(work.ubuff1, work.lambda, work.Ds, data.l, data.q)
    nt_multiply_W_from!(work.ubuff2, work.xyz, Dz_offset, data, work)
    @inbounds for i in eachindex(work.ubuff3)
        work.ubuff3[i] = work.ubuff1[i] - work.ubuff2[i]
    end
    nt_multiply_W!(work.Ds, work.ubuff3, data, work)

    a = min(linesearch!(solver, work.s, work.Ds, T(0.99)), linesearch_from!(solver, work.z, work.xyz, Dz_offset, T(0.99)))
    work.a = a

    add_scaled_from!(work.x, a, work.xyz, 1, data.n)
    add_scaled!(work.s, a, work.Ds)
    add_scaled_from!(work.y, a, work.xyz, data.n + 1, data.p)
    add_scaled_from!(work.z, a, work.xyz, Dz_offset, data.m)
    return nothing
end

function check_stopping!(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    scaling = solver.scaling

    binf = data.p > 0 ? weighted_inf_norm(data.b, scaling.Einvruiz) : zero(T)

    sinf = data.m > 0 ? weighted_inf_norm(work.s, scaling.Fruiz) : zero(T)

    cinf = weighted_inf_norm(work.x, scaling.Dinvruiz)

    hinf = data.m > 0 ? weighted_inf_norm(data.h, scaling.Finvruiz) : zero(T)

    eq_res = data.p > 0 ? weighted_inf_norm_from(work.kktres, data.n + 1, scaling.Einvruiz, data.p) : zero(T)

    conic_res = data.m > 0 ? weighted_inf_norm_from(work.kktres, data.n + data.p + 1, scaling.Finvruiz, data.m) : zero(T)
    solver.solution.pres = max(eq_res, conic_res)

    solver.solution.dres = weighted_inf_norm_from(work.kktres, 1, scaling.Dinvruiz, data.n) * scaling.kinv

    if data.m > 0
        solver.solution.gap = weighted_dot(work.s, work.z, scaling.Fruiz) * scaling.kinv
    else
        solver.solution.gap = zero(T)
    end

    pres_rel = max(max(work.Axinf, binf), max(max(work.Gxinf, hinf), sinf))
    dres_rel = max(max(work.Pxinf, work.Atyinf), max(work.Gtzinf, cinf)) * scaling.kinv
    ctx = dot(data.c, work.x)
    bty = dot(data.b, work.y)
    htz = dot(data.h, work.z)
    pobj = abs(T(0.5) * work.xPx + ctx)
    dobj = abs(-T(0.5) * work.xPx - bty - htz)
    gap_rel = max(one(T), max(pobj, dobj))

    if work.a < T(1e-8)
        if solver.solution.pres < solver.settings.abstol_inacc + solver.settings.reltol_inacc * pres_rel &&
           solver.solution.dres < solver.settings.abstol_inacc + solver.settings.reltol_inacc * dres_rel &&
           solver.solution.gap < solver.settings.abstol_inacc + solver.settings.reltol_inacc * gap_rel
            solver.solution.status = QOCO_SOLVED_INACCURATE
            solver.solution.status_detail = "stalled step but met inaccurate tolerances"
        else
            solver.solution.status = QOCO_NUMERICAL_ERROR
            solver.solution.status_detail = "step length dropped below 1e-8"
        end
        return true
    end

    if solver.solution.pres < solver.settings.abstol + solver.settings.reltol * pres_rel &&
       solver.solution.dres < solver.settings.abstol + solver.settings.reltol * dres_rel &&
       solver.solution.gap < solver.settings.abstol + solver.settings.reltol * gap_rel
        solver.solution.status = QOCO_SOLVED
        solver.solution.status_detail = ""
        return true
    end
    return false
end
