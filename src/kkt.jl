# Whether a cone of this dimension uses the sparse expansion.
@inline soc_is_expanded(q::Integer, threshold::Integer) = q >= threshold

# Storage for the Nesterov-Todd values that live in the KKT matrix. A dense
# cone needs its whole upper triangle; an expanded cone needs only its q
# diagonal entries here, with the rest in `soc_aux`.
function kkt_nt_nnz(l::Integer, qdims::AbstractVector{<:Integer}, expanded::AbstractVector{Bool})
    total = Int(l)
    for (block, q) in enumerate(qdims)
        total += expanded[block] ? Int(q) : Int(q) * (Int(q) + 1) ÷ 2
    end
    return total
end

kkt_nt_nnz(l::Integer, qdims::AbstractVector{<:Integer}) =
    kkt_nt_nnz(l, qdims, fill(false, length(qdims)))

# Two auxiliary columns per expanded cone: q coupling entries plus one
# diagonal each.
function kkt_aux_nnz(qdims::AbstractVector{<:Integer}, expanded::AbstractVector{Bool})
    total = 0
    for (block, q) in enumerate(qdims)
        expanded[block] && (total += 2 * Int(q) + 2)
    end
    return total
end

count_expanded(expanded::AbstractVector{Bool}) = count(expanded)

function construct_kkt(data::ProblemData{T,Ti}, settings::Settings{T}, work::Workspace{T,Ti}) where {T<:AbstractFloat,Ti<:Integer}
    n = Int(data.n)
    p = Int(data.p)
    m = Int(data.m)
    naux = 2 * count_expanded(work.soc_expanded)
    N = n + p + m + naux
    Wnnz = length(work.WtW)
    Anzz = length(work.soc_aux)
    total_nnz = nnz(data.P) + nnz(data.A) + nnz(data.G) + Wnnz + p + Anzz

    colptr = Vector{Ti}(undef, N + 1)
    rowval = Vector{Ti}(undef, total_nnz)
    nzval = Vector{T}(undef, total_nnz)
    nt2kkt = Vector{Ti}(undef, Wnnz)
    aux2kkt = Vector{Ti}(undef, Anzz)
    auxpos_diag = Vector{Ti}()
    auxneg_diag = Vector{Ti}()
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
    for (block, q) in enumerate(data.q)
        expanded = work.soc_expanded[block]
        for global_col in cone_start:(cone_start + q - 1)
            for k in data.Gt.colptr[global_col]:(data.Gt.colptr[global_col + 1] - 1)
                rowval[nz] = data.Gt.rowval[k]
                nzval[nz] = data.Gt.nzval[k]
                Gt2kkt[k] = Ti(nz)
                nz += 1
            end
            local_col = global_col - cone_start + 1
            if expanded
                # Only the diagonal lives in the cone column; the rank-two
                # part moves into the auxiliary columns below.
                rowval[nz] = Ti(n + p + global_col)
                nzval[nz] = -one(T)
                nt2kkt[ntpos] = Ti(nz)
                push!(ntdiag_positions, Ti(ntpos))
                ntpos += 1
                nz += 1
            else
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
            end
            colptr[col + 1] = Ti(nz)
            col += 1
        end
        cone_start += q
    end

    # Auxiliary columns, two per expanded cone. Column 2i-1 couples the cone
    # rows to the first auxiliary variable and carries the diagonal +1; column
    # 2i couples to the second and carries -1.
    auxpos = 1
    aux_variable = n + p + m
    for (block, q) in enumerate(data.q)
        work.soc_expanded[block] || continue
        first_row = n + p + Int(work.soc_offsets[block])
        for which in 1:2
            aux_variable += 1
            for local_row in 1:q
                rowval[nz] = Ti(first_row + local_row - 1)
                nzval[nz] = zero(T)
                aux2kkt[auxpos] = Ti(nz)
                auxpos += 1
                nz += 1
            end
            rowval[nz] = Ti(aux_variable)
            nzval[nz] = which == 1 ? one(T) : -one(T)
            aux2kkt[auxpos] = Ti(nz)
            push!(which == 1 ? auxpos_diag : auxneg_diag, Ti(auxpos))
            auxpos += 1
            nz += 1
            colptr[col + 1] = Ti(nz)
            col += 1
        end
    end

    K = SparseMatrixCSC(N, N, colptr, rowval, nzval)
    return K, nt2kkt, ntdiag_positions, aux2kkt, auxpos_diag, auxneg_diag, P2kkt, At2kkt, Gt2kkt
end

# Rank-two decomposition of the scaled Nesterov-Todd block of one cone.
#
# With W = scale*(2 v v' - J) and v'Jv = 1, expanding the square gives
#
#     W'W = scale^2 ( I + U M U' ),   U = [v  Jv],  M = [4 v'v  -2; -2  0].
#
# M is 2x2 with determinant -4, so it has exactly one positive and one negative
# eigenvalue, and splitting it into that pair of rank-one terms turns the dense
# block into
#
#     W'W = scale^2 ( I + g g' - f f' ).
#
# Because g and f are combinations of v and Jv, and Jv only flips the tail sign,
# each of them is two scalars times the head and tail of v: O(q) work and O(q)
# storage instead of O(q^2). The negative eigenvalue is recovered from the
# determinant rather than from the subtraction, which would cancel for large
# v'v.
function soc_expansion_vectors!(work::Workspace{T}, block::Int, q::Int) where {T<:AbstractFloat}
    idx = Int(work.soc_offsets[block])
    aoffset = Int(work.soc_aux_offsets[block])
    scale = work.nt_scale[block]
    inverse_scale2 = positive_inv(scale * scale)
    vv = zero(T)
    @inbounds for k in 0:(q - 1)
        value = work.nt_v[idx + k]
        vv += value * value
    end
    alpha = T(4) * vv
    lambda_plus = T(0.5) * (alpha + sqrt(alpha * alpha + T(16)))
    # lambda_plus * lambda_minus = det(M) = -4.
    lambda_minus = -T(4) / lambda_plus

    # Eigenvector of M for eigenvalue lambda is proportional to (-lambda, 2).
    norm_plus = hypot(lambda_plus, T(2))
    norm_minus = hypot(lambda_minus, T(2))
    root_plus = sqrt(lambda_plus)
    root_minus = sqrt(-lambda_minus)
    p1 = root_plus * (-lambda_plus / norm_plus)
    p2 = root_plus * (T(2) / norm_plus)
    n1 = root_minus * (-lambda_minus / norm_minus)
    n2 = root_minus * (T(2) / norm_minus)

    # g = p1 v + p2 Jv and f = n1 v + n2 Jv, and Jv negates the tail.
    g_head = p1 + p2
    g_tail = p1 - p2
    f_head = n1 + n2
    f_tail = n1 - n2

    # An auxiliary variable can be rescaled freely: with coupling kappa*g the
    # matching diagonal is kappa^2/scale^2, and the Schur complement is the
    # same for every kappa. The choice is not numerically free, though. The
    # auxiliary pivot is perturbed by the static regularization and tested
    # against the dynamic one, both of which are absolute, so a pivot that
    # drifts away from unit size either has its rank-one term perturbed by a
    # large relative amount or is replaced outright. Taking kappa = scale
    # pins both pivots to exactly +1 and -1 for the life of the solve, which
    # leaves only the off-diagonal entries to grow.
    scaled_head = scale * work.nt_v[idx]
    @inbounds begin
        work.soc_aux[aoffset] = g_head * scaled_head
        for k in 1:(q - 1)
            work.soc_aux[aoffset + k] = scale * g_tail * work.nt_v[idx + k]
        end
        work.soc_aux[aoffset + q] = one(T)
        work.soc_aux[aoffset + q + 1] = f_head * scaled_head
        for k in 1:(q - 1)
            work.soc_aux[aoffset + q + 1 + k] = scale * f_tail * work.nt_v[idx + k]
        end
        work.soc_aux[aoffset + 2 * q + 1] = -one(T)
    end
    return inverse_scale2
end

function set_identity_scalings!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    fill!(work.WtW, zero(T))
    @inbounds for i in 1:Int(data.l)
        work.WtW[i] = one(T)
    end
    for (block, q) in enumerate(data.q)
        toffset = work.Wtri_offsets[block]
        idx = work.soc_offsets[block]
        work.nt_scale[block] = one(T)
        work.nt_v[idx] = one(T)
        @inbounds for k in 1:(q - 1)
            work.nt_v[idx + k] = zero(T)
        end
        if work.soc_expanded[block]
            @inbounds for j in 1:q
                work.WtW[toffset + j - 1] = one(T)
            end
            soc_expansion_vectors!(work, block, Int(q))
        else
            @inbounds for j in 1:q
                work.WtW[toffset + (j * (j - 1)) ÷ 2 + j - 1] = one(T)
            end
        end
    end
    return nothing
end

const MAX_FACTOR_RETRIES = 3
const MAX_REGULARIZE_EPS = 1e-4
const MAX_REGULARIZE_DELTA = 1e-2

function update_nt_block!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    work = solver.work
    linsys = solver.linsys
    reg = solver.settings.kkt_static_reg
    @inbounds for i in eachindex(work.WtW, linsys.nt_values)
        linsys.nt_values[i] = -work.WtW[i]
    end
    @inbounds for pos in linsys.ntdiag_positions
        linsys.nt_values[pos] -= reg
    end
    QDLDL.update_values_internal!(linsys.factor, linsys.nt2kkt, linsys.nt_values)
    if !isempty(linsys.aux_values)
        # The expanded-cone entries go in unnegated. The two auxiliary
        # diagonals per cone sit on opposite sides of the quasidefinite
        # partition, so they are regularized in opposite directions.
        @inbounds for i in eachindex(work.soc_aux, linsys.aux_values)
            linsys.aux_values[i] = work.soc_aux[i]
        end
        @inbounds for pos in linsys.auxpos_diag
            linsys.aux_values[pos] += reg
        end
        @inbounds for pos in linsys.auxneg_diag
            linsys.aux_values[pos] -= reg
        end
        QDLDL.update_values_internal!(linsys.factor, linsys.aux2kkt, linsys.aux_values)
    end
    factor = linsys.factor
    profile = solver.solution.profile
    # Escalating static regularization is a response to a numerically failed
    # factorization only. An interrupt, or a genuine programming error, must
    # propagate instead of being retried into a different regularization.
    for attempt in 0:MAX_FACTOR_RETRIES
        try
            QDLDL.refactor!(factor)
            break
        catch error
            error isa QDLDL.FactorizationFailure || rethrow(error)
            attempt == MAX_FACTOR_RETRIES && rethrow(error)
            factor.workspace.regularize_eps =
                min(factor.workspace.regularize_eps * T(10), T(MAX_REGULARIZE_EPS))
            factor.workspace.regularize_delta =
                min(factor.workspace.regularize_delta * T(10), T(MAX_REGULARIZE_DELTA))
            profile.factorization_retries += 1
            profile.dynamic_regularizations += 1
        end
    end
    profile.regularized_pivots += Int(QDLDL.regularized_entries(factor))
    profile.nt_refactors += 1
    return nothing
end

@inline _mul_P!(y, data, x) = mul_upper_symmetric!(y, data.P, x)
@inline _mul_A!(y, data, x) = mul!(y, data.A, x)
@inline _mul_At!(y, data, x) = mul!(y, data.At, x)
@inline _mul_G!(y, data, x) = mul!(y, data.G, x)
@inline _mul_Gt!(y, data, x) = mul!(y, data.Gt, x)

# Product with the *original* Newton matrix
#
#     K = [ P    A'    G'    ]
#         [ A    0     0     ]
#         [ G    0    -W'W   ]
#
# where P is the mathematical Hessian, that is, `data.P` with the static
# regularization shift removed. This is deliberately neither the matrix that
# was factorized (which additionally carries -reg on the equality and cone
# diagonals and +reg on the primal diagonal) nor an approximation of it: the
# factorization of the regularized matrix is used as a preconditioner, while
# iterative refinement measures its residual against this operator, so that
# refinement converges to the solution of the unregularized Newton system.
function kkt_multiply!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    data::ProblemData{T},
    work::Workspace{T},
    static_reg::T;
    include_nt::Bool = true,
) where {T<:AbstractFloat}
    n = Int(data.n)
    p = Int(data.p)
    m = Int(data.m)

    xpr = view(x, 1:n)
    ypr = view(y, 1:n)
    _mul_P!(ypr, data, xpr)
    add_scaled!(ypr, -static_reg, xpr)

    if p > 0
        yeq = view(y, (n + 1):(n + p))
        _mul_At!(work.xbuff, data, view(x, (n + 1):(n + p)))
        add_scaled!(ypr, one(T), work.xbuff)
        _mul_A!(work.ybuff, data, xpr)
        copyto!(yeq, work.ybuff)
    end

    if m > 0
        ycon = view(y, (n + p + 1):(n + p + m))
        _mul_Gt!(work.xbuff, data, view(x, (n + p + 1):(n + p + m)))
        add_scaled!(ypr, one(T), work.xbuff)
        _mul_G!(work.ubuff1, data, xpr)
        copyto!(ycon, work.ubuff1)
        if include_nt
            nt_multiply_W!(work.ubuff1, view(x, (n + p + 1):(n + p + m)), data, work)
            nt_multiply_W!(work.ubuff2, work.ubuff1, data, work)
            add_scaled!(ycon, -one(T), work.ubuff2)
        end
    end
    return y
end

# Apply the factorization to a right-hand side expressed in the original
# variables. When second-order cones have been expanded the factorized system
# is larger, so the right-hand side is padded with zeros for the auxiliary
# rows and only the leading part of the answer is kept. Eliminating those
# auxiliary variables reproduces the original Newton block exactly, so this is
# a solve of the same system, not of an approximation to it.
function apply_factor!(
    solver::CoreSolver{T},
    rhs::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:AbstractFloat}
    augmented = solver.work.xyz_aug
    if isempty(augmented)
        x === rhs || copyto!(x, rhs)
        QDLDL.solve!(solver.linsys.factor, x)
        return x
    end
    original = length(rhs)
    copyto!(augmented, 1, rhs, 1, original)
    @inbounds for i in (original + 1):length(augmented)
        augmented[i] = zero(T)
    end
    QDLDL.solve!(solver.linsys.factor, augmented)
    copyto!(x, 1, augmented, 1, original)
    return x
end

function _solve_linsys_fast!(
    solver::CoreSolver{T},
    rhs::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:AbstractFloat}
    apply_factor!(solver, rhs, x)
    solver.settings.iter_ref_iters == 0 && return x
    rhs_norm = max(one(T), inf_norm(rhs))
    for _ in 1:solver.settings.iter_ref_iters
        kkt_multiply!(solver.work.xyzbuff1, x, solver.data, solver.work, solver.settings.kkt_static_reg)
        @inbounds for i in eachindex(x)
            solver.work.xyzbuff1[i] = rhs[i] - solver.work.xyzbuff1[i]
        end
        inf_norm(solver.work.xyzbuff1) <= solver.settings.iter_ref_tol * rhs_norm && break
        apply_factor!(solver, solver.work.xyzbuff1, solver.work.xyzbuff1)
        add_scaled!(x, one(T), solver.work.xyzbuff1)
    end
    return x
end

function _solve_linsys_profiled!(
    solver::CoreSolver{T},
    rhs::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:AbstractFloat}
    profile = solver.solution.profile
    tsolve = time_ns()
    apply_factor!(solver, rhs, x)
    profile.linsys_solve_time_sec += elapsed_time_sec(tsolve)
    profile.linsys_solves += 1
    solver.settings.iter_ref_iters == 0 && return x
    rhs_norm = max(one(T), inf_norm(rhs))
    for _ in 1:solver.settings.iter_ref_iters
        kkt_multiply!(solver.work.xyzbuff1, x, solver.data, solver.work, solver.settings.kkt_static_reg)
        @inbounds for i in eachindex(x)
            solver.work.xyzbuff1[i] = rhs[i] - solver.work.xyzbuff1[i]
        end
        inf_norm(solver.work.xyzbuff1) <= solver.settings.iter_ref_tol * rhs_norm && break
        trefine = time_ns()
        apply_factor!(solver, solver.work.xyzbuff1, solver.work.xyzbuff1)
        profile.linsys_refine_time_sec += elapsed_time_sec(trefine)
        profile.linsys_refinements += 1
        add_scaled!(x, one(T), solver.work.xyzbuff1)
    end
    return x
end

@inline function solve_linsys!(
    solver::CoreSolver{T},
    rhs::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:AbstractFloat}
    return solver.settings.profile ?
           _solve_linsys_profiled!(solver, rhs, x) :
           _solve_linsys_fast!(solver, rhs, x)
end

function initialize_ipm!(solver::CoreSolver{T}) where {T<:AbstractFloat}
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

# Norms of the problem data in original units. These do not change during a
# solve, so they are computed once and reused by every stopping check.
function refresh_data_norms!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    scaling = solver.scaling
    # c_hat = k D c, so |c|_inf = |D^-1 c_hat|_inf / k.
    work.cinf = weighted_inf_norm(data.c, scaling.Dinvruiz) * scaling.kinv
    work.binf = data.p > 0 ? weighted_inf_norm(data.b, scaling.Einvruiz) : zero(T)
    work.hinf = data.m > 0 ? weighted_inf_norm(data.h, scaling.Finvruiz) : zero(T)
    return nothing
end

# Residual of the current iterate. `work.kktres` stays in scaled coordinates
# so that no unscaled vector has to be allocated per iteration; the reference
# norms stored alongside it are converted to original units immediately, using
# the identities documented at the top of equilibration.jl.
function compute_kkt_residual!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    scaling = solver.scaling
    kinv = scaling.kinv
    _mul_P!(work.xbuff, data, work.x)
    # data.P carries the static regularization shift; the mathematical
    # objective and the original KKT residual must exclude it.
    add_scaled!(work.xbuff, -solver.settings.kkt_static_reg, work.x)
    work.quad_obj = dot(work.xbuff, work.x)
    # x'Px = dot(x_hat, P_hat x_hat) / k.
    work.xPx = work.quad_obj * kinv
    # P_hat x_hat = k D (P x), so |P x|_inf = |D^-1 P_hat x_hat|_inf / k.
    work.Pxinf = weighted_inf_norm(work.xbuff, scaling.Dinvruiz) * kinv
    copyto!(work.kktres, 1, work.xbuff, 1, data.n)
    add_scaled_to!(work.kktres, 1, one(T), data.c, data.n)

    if data.p > 0
        _mul_At!(work.xbuff, data, work.y)
        work.Atyinf = weighted_inf_norm(work.xbuff, scaling.Dinvruiz) * kinv
        add_scaled_to!(work.kktres, 1, one(T), work.xbuff, data.n)
        _mul_A!(work.ybuff, data, work.x)
        work.Axinf = weighted_inf_norm(work.ybuff, scaling.Einvruiz)
        copyto!(work.kktres, data.n + 1, work.ybuff, 1, data.p)
        add_scaled_to!(work.kktres, data.n + 1, -one(T), data.b, data.p)
    else
        work.Atyinf = zero(T)
        work.Axinf = zero(T)
    end

    if data.m > 0
        _mul_Gt!(work.xbuff, data, work.z)
        work.Gtzinf = weighted_inf_norm(work.xbuff, scaling.Dinvruiz) * kinv
        add_scaled_to!(work.kktres, 1, one(T), work.xbuff, data.n)
        _mul_G!(work.ubuff1, data, work.x)
        work.Gxinf = weighted_inf_norm(work.ubuff1, scaling.Finvruiz)
        copyto!(work.kktres, data.n + data.p + 1, work.ubuff1, 1, data.m)
        add_scaled_to!(work.kktres, data.n + data.p + 1, -one(T), data.h, data.m)
        add_scaled_to!(work.kktres, data.n + data.p + 1, one(T), work.s, data.m)
        # s = F^-1 s_hat.
        work.sinf = weighted_inf_norm(work.s, scaling.Finvruiz)
    else
        work.Gtzinf = zero(T)
        work.Gxinf = zero(T)
        work.sinf = zero(T)
    end
    return nothing
end

function compute_mu!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    m = length(solver.work.s)
    solver.work.mu = m == 0 ? zero(T) : dot(solver.work.s, solver.work.z) / T(m)
    return solver.work.mu
end

function compute_objective!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    # dot(c_hat, x_hat) = k c'x and dot(x_hat, P_hat x_hat) = k x'Px.
    obj = dot(work.x, data.c) + T(0.5) * work.quad_obj
    solver.solution.obj = obj * solver.scaling.kinv
    return solver.solution.obj
end

function construct_kkt_aff_rhs!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    copy_negate!(work.rhs, work.kktres)
    nt_multiply_W!(work.ubuff1, work.lambda, data, work)
    add_scaled_to!(work.rhs, data.n + data.p + 1, one(T), work.ubuff1, data.m)
    return nothing
end

function construct_kkt_comb_rhs!(solver::CoreSolver{T}) where {T<:AbstractFloat}
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

function compute_centering!(solver::CoreSolver{T}) where {T<:AbstractFloat}
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
    # The centering ratio needs its sign, so it cannot use the
    # no-restriction sentinel. A complementarity that has already collapsed
    # leaves the ratio undefined; the safe reading is full centering.
    rho = checked_div(dot(work.ubuff1, work.ubuff2), dot(work.z, work.s))
    sigma = isnan(rho) ? one(T) : clamp(rho, zero(T), one(T))
    work.sigma = sigma * sigma * sigma
    return work.sigma
end

function predictor_corrector!(solver::CoreSolver{T}) where {T<:AbstractFloat}
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

    # An infinite search direction is as unusable as a NaN one.
    if has_nonfinite(work.xyz)
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

# Relative violation of cone membership for one vector, normalized by the
# magnitude of the vector so that the test means the same thing at any scale.
# Membership is preserved by the row scaling (the orthant scales are positive
# and each second-order cone carries one common scale), so this can be
# evaluated on the internal iterate.
function cone_violation(u::AbstractVector{T}, l::Integer, qdims::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    isempty(u) && return zero(T)
    all_finite(u) || return T(Inf)
    scale = max(one(T), inf_norm(u))
    return max(zero(T), cone_residual(u, Int(l), qdims)) / scale
end

# Reference magnitudes for the relative stopping tests, all in original units.
# `p_ref` and `d_ref` follow the usual conic reference norms; `g_ref` is an
# explicitly chosen objective-unit reference. The expression
# `-0.5 x'Px - b'y - h'z` used in `g_ref` is a *scale*, not a certified dual
# bound: it is not valid as a lower bound at an arbitrary nonstationary
# iterate, and no dual objective is reported from it.
function _stopping_references(solver::CoreSolver{T}) where {T<:AbstractFloat}
    work = solver.work
    p_ref = max(work.Axinf, work.binf, work.Gxinf, work.hinf, work.sinf)
    d_ref = max(work.Pxinf, work.cinf, work.Atyinf, work.Gtzinf)
    kinv = solver.scaling.kinv
    ctx = dot(solver.data.c, work.x) * kinv
    bty = solver.data.p > 0 ? dot(solver.data.b, work.y) * kinv : zero(T)
    htz = solver.data.m > 0 ? dot(solver.data.h, work.z) * kinv : zero(T)
    pobj = T(0.5) * work.xPx + ctx
    dobj = -T(0.5) * work.xPx - bty - htz
    g_ref = max(one(T), abs(pobj), abs(dobj))
    return p_ref, d_ref, g_ref
end

# The single accuracy measure used for regular stopping, inaccurate stopping,
# best-iterate ranking and final status assessment: the largest of the three
# residual-to-tolerance ratios. A value of at most one means every requested
# tolerance is satisfied. A nonfinite or cone-invalid iterate scores `Inf`, so
# it can never be mistaken for a good one.
function solution_quality(
    solver::CoreSolver{T},
    abstol::T,
    reltol::T,
) where {T<:AbstractFloat}
    sol = solver.solution
    (isfinite(sol.pres) && isfinite(sol.dres) && isfinite(sol.gap)) || return T(Inf)
    sol.cone_valid || return T(Inf)
    p_ref, d_ref, g_ref = _stopping_references(solver)
    ratio_p = sol.pres / (abstol + reltol * p_ref)
    ratio_d = sol.dres / (abstol + reltol * d_ref)
    ratio_g = abs(sol.gap) / (abstol + reltol * g_ref)
    quality = max(ratio_p, max(ratio_d, ratio_g))
    return isfinite(quality) ? quality : T(Inf)
end

# Tolerance below which a cone violation is treated as rounding rather than an
# invalid point.
_cone_tolerance(::Type{T}) where {T<:AbstractFloat} = T(16) * sqrt(eps(T))

# Fill in every reported metric for the current iterate. Callers must have run
# `compute_kkt_residual!` first; `check_stopping!` and the finalization path
# both go through here so that reported metrics can never describe a different
# iterate from the returned vectors.
function assess_iterate!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    work = solver.work
    scaling = solver.scaling
    sol = solver.solution

    eq_res = data.p > 0 ?
             weighted_inf_norm_from(work.kktres, data.n + 1, scaling.Einvruiz, data.p) :
             zero(T)
    conic_res = data.m > 0 ?
                weighted_inf_norm_from(work.kktres, data.n + data.p + 1, scaling.Finvruiz, data.m) :
                zero(T)
    sol.pres = max(eq_res, conic_res)
    sol.dres = weighted_inf_norm_from(work.kktres, 1, scaling.Dinvruiz, data.n) * scaling.kinv
    # s'z = dot(s_hat, z_hat) / k. Neither vector may be reweighted here: the
    # row scale on s is the reciprocal of the one on z, so they cancel.
    sol.gap = data.m > 0 ? dot(work.s, work.z) * scaling.kinv : zero(T)

    tolerance = _cone_tolerance(T)
    sol.cone_valid =
        all_finite(work.x) && all_finite(work.y) &&
        cone_violation(work.s, data.l, data.q) <= tolerance &&
        cone_violation(work.z, data.l, data.q) <= tolerance

    compute_objective!(solver)
    sol.quality = solution_quality(solver, solver.settings.abstol, solver.settings.reltol)
    return sol.quality
end

function check_stopping!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    settings = solver.settings
    sol = solver.solution
    assess_iterate!(solver)
    quality_inacc = solution_quality(solver, settings.abstol_inacc, settings.reltol_inacc)

    # A nonfinite residual is an outright failure, not a large one. Stop and
    # say so rather than iterating on NaN until the iteration limit and then
    # reporting that limit as the reason.
    if !(isfinite(sol.pres) && isfinite(sol.dres) && isfinite(sol.gap))
        sol.status = QOCO_NUMERICAL_ERROR
        sol.status_detail = "residuals are not finite"
        return true
    end

    if solver.work.a < default_min_step(T)
        if quality_inacc <= one(T)
            sol.status = QOCO_SOLVED_INACCURATE
            sol.status_detail = "stalled step but met inaccurate tolerances"
        else
            sol.status = QOCO_NUMERICAL_ERROR
            sol.status_detail = "step length dropped below 1e-8"
        end
        return true
    end

    if sol.quality <= one(T)
        sol.status = QOCO_SOLVED
        sol.status_detail = ""
        return true
    end
    return false
end
