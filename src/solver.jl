function _soc_offsets(
    l::Integer,
    qdims::AbstractVector{Ti},
    expanded::AbstractVector{Bool},
) where {Ti<:Integer}
    soc_offsets = Vector{Ti}(undef, length(qdims))
    Wtri_offsets = Vector{Ti}(undef, length(qdims))
    aux_offsets = Vector{Ti}(undef, length(qdims))
    soff = Ti(l + 1)
    toff = Ti(l + 1)
    aoff = one(Ti)
    @inbounds for i in eachindex(qdims)
        q = qdims[i]
        soc_offsets[i] = soff
        Wtri_offsets[i] = toff
        aux_offsets[i] = aoff
        soff += q
        if expanded[i]
            toff += q
            aoff += 2 * q + 2
        else
            toff += q * (q + 1) ÷ 2
        end
    end
    return soc_offsets, Wtri_offsets, aux_offsets
end

function _workspace(data::ProblemData{T,Ti}, settings::Settings{T}) where {T<:AbstractFloat,Ti<:Integer}
    expanded = Bool[
        soc_is_expanded(q, settings.soc_expansion_threshold) for q in data.q
    ]
    Wnnz = kkt_nt_nnz(data.l, data.q, expanded)
    auxnnz = kkt_aux_nnz(data.q, expanded)
    naux = 2 * count_expanded(expanded)
    maxq = isempty(data.q) ? 0 : maximum(data.q)
    soc_offsets, Wtri_offsets, aux_offsets = _soc_offsets(data.l, data.q, expanded)
    return Workspace{T,Ti}(
        zeros(T, data.n),
        zeros(T, data.m),
        zeros(T, data.p),
        zeros(T, data.m),
        zero(T),
        one(T),
        zero(T),
        zeros(T, Wnnz),
        zeros(T, data.m),
        ones(T, length(data.q)),
        zeros(T, data.m),
        zeros(T, maxq),
        zeros(T, maxq),
        zeros(T, data.n),
        zeros(T, data.p),
        zeros(T, data.m),
        zeros(T, data.m),
        zeros(T, data.m),
        zeros(T, data.m),
        zeros(T, data.n + data.p + data.m),
        zeros(T, data.n + data.p + data.m),
        zeros(T, data.n + data.p + data.m),
        zeros(T, data.n + data.p + data.m),
        zeros(T, data.n + data.p + data.m),
        soc_offsets,
        Wtri_offsets,
        expanded,
        zeros(T, auxnnz),
        aux_offsets,
        naux == 0 ? T[] : zeros(T, data.n + data.p + data.m + naux),
        zero(T), # quad_obj
        zero(T), # xPx
        zero(T), # Pxinf
        zero(T), # Atyinf
        zero(T), # Gtzinf
        zero(T), # Axinf
        zero(T), # Gxinf
        zero(T), # sinf
        zero(T), # cinf
        zero(T), # binf
        zero(T), # hinf
    )
end

function _problem_data(
    P::Union{Nothing,SparseMatrixCSC{T,Ti}},
    c::AbstractVector{T},
    A::Union{Nothing,SparseMatrixCSC{T,Ti}},
    b::Union{Nothing,AbstractVector{T}},
    G::Union{Nothing,SparseMatrixCSC{T,Ti}},
    h::Union{Nothing,AbstractVector{T}},
    l::Integer,
    q::AbstractVector{<:Integer},
    settings::Settings{T},
) where {T<:AbstractFloat,Ti<:Integer}
    validate_data(P, c, A, b, G, h, l, q)
    n = length(c)
    A0 = A === nothing ? spzeros(T, 0, n) : copy(A)
    G0 = G === nothing ? spzeros(T, 0, n) : copy(G)
    b0 = b === nothing ? zeros(T, 0) : collect(b)
    h0 = h === nothing ? zeros(T, 0) : collect(h)
    P0 = P === nothing ? spzeros(T, n, n) : begin
        istriu(P) || throw(ArgumentError("P must use an upper-triangular CSC convention"))
        copy(P)
    end
    qv = Ti.(collect(q))

    if settings.convexity_check != :none && nnz(P0) > 0
        is_positive_semidefinite(P0, zero(T), settings.convexity_dense_limit) ||
            throw(ArgumentError("quadratic objective matrix must be positive semidefinite"))
    end

    At0, AtoAt = create_transposed_matrix_with_map(A0)
    Gt0, GtoGt = create_transposed_matrix_with_map(G0)
    AfromAt = inverse_entry_map(AtoAt)
    GfromGt = inverse_entry_map(GtoGt)
    P1, Padded_idx = regularize_P_with_info(P0, zero(T))
    data = ProblemData{T,Ti}(
        P1,
        entry_columns(P1),
        collect(c),
        A0,
        entry_columns(A0),
        At0,
        AtoAt,
        AfromAt,
        b0,
        G0,
        entry_columns(G0),
        Gt0,
        GtoGt,
        GfromGt,
        h0,
        Int(l),
        qv,
        n,
        length(h0),
        length(b0),
        Padded_idx,
        ScalingStats(T),
        false,
    )
    scaling = initialize_scaling(data)
    ruiz_equilibration!(
        data,
        scaling,
        settings.scaling_mode == :none ? 0 : settings.ruiz_iters,
    )
    regularize_existing_P!(data.P, settings.kkt_static_reg)
    return data, scaling
end

function _refresh_scaling_stats!(solver::CoreSolver)
    data = solver.data
    if data.stats_dirty
        data.stats = compute_scaling_statistics(data)
        data.stats_dirty = false
    end
    return data.stats
end

function _fill_static_values!(dest::AbstractVector{T}, data::ProblemData{T}) where {T<:AbstractFloat}
    pos = 1
    np = nnz(data.P)
    if np > 0
        copyto!(view(dest, pos:(pos + np - 1)), data.P.nzval)
        pos += np
    end
    nat = nnz(data.At)
    if nat > 0
        copyto!(view(dest, pos:(pos + nat - 1)), data.At.nzval)
        pos += nat
    end
    ngt = nnz(data.Gt)
    if ngt > 0
        copyto!(view(dest, pos:(pos + ngt - 1)), data.Gt.nzval)
    end
    return dest
end

function _linsys(data::ProblemData{T,Ti}, settings::Settings{T}, work::Workspace{T,Ti}) where {T<:AbstractFloat,Ti<:Integer}
    if data.n + data.p + data.m == 0
        return LinearSystem{T,Ti}(
            nothing, Ti[], Ti[], zeros(T, length(work.WtW)),
            Ti[], T[], Ti[], Ti[], Ti[], T[],
        )
    end
    K, nt2kkt, ntdiag_positions, aux2kkt, auxpos_diag, auxneg_diag,
    P2kkt, At2kkt, Gt2kkt = construct_kkt(data, settings, work)
    # The augmented system stays quasidefinite: the positive block holds the
    # primal variables and the first auxiliary variable of each expanded cone,
    # the negative block holds everything else.
    signs = vcat(ones(Ti, data.n), -ones(Ti, data.p + data.m))
    for _ in 1:count_expanded(work.soc_expanded)
        push!(signs, one(Ti))
        push!(signs, -one(Ti))
    end
    factor = QDLDL.qdldl(
        K;
        Dsigns = signs,
        regularize_eps = settings.kkt_dynamic_reg,
        regularize_delta = settings.kkt_dynamic_reg,
    )
    static2kkt = Vector{Ti}(undef, length(P2kkt) + length(At2kkt) + length(Gt2kkt))
    pos = 1
    copyto!(view(static2kkt, pos:(pos + length(P2kkt) - 1)), P2kkt)
    pos += length(P2kkt)
    copyto!(view(static2kkt, pos:(pos + length(At2kkt) - 1)), At2kkt)
    pos += length(At2kkt)
    copyto!(view(static2kkt, pos:(pos + length(Gt2kkt) - 1)), Gt2kkt)
    nt2kkt = QDLDL.map_indices(factor, nt2kkt)
    aux2kkt = QDLDL.map_indices(factor, aux2kkt)
    static2kkt = QDLDL.map_indices(factor, static2kkt)
    static_values = zeros(T, length(static2kkt))
    _fill_static_values!(static_values, data)
    return LinearSystem{T,Ti}(
        factor,
        nt2kkt,
        ntdiag_positions,
        zeros(T, length(work.WtW)),
        aux2kkt,
        zeros(T, length(work.soc_aux)),
        auxpos_diag,
        auxneg_diag,
        static2kkt,
        static_values,
    )
end

function CoreSolver(
    P::Union{Nothing,SparseMatrixCSC{T,Ti}},
    c::AbstractVector{T},
    A::Union{Nothing,SparseMatrixCSC{T,Ti}},
    b::Union{Nothing,AbstractVector{T}},
    G::Union{Nothing,SparseMatrixCSC{T,Ti}},
    h::Union{Nothing,AbstractVector{T}},
    l::Integer,
    q::AbstractVector{<:Integer};
    settings::Settings{T} = default_settings(T),
) where {T<:AbstractFloat,Ti<:Integer}
    validate_settings(settings)
    t0 = time_ns()
    tphase = time_ns()
    data, scaling = _problem_data(P, c, A, b, G, h, l, q, settings)
    problem_data_time_sec = elapsed_time_sec(tphase)
    tphase = time_ns()
    work = _workspace(data, settings)
    workspace_time_sec = elapsed_time_sec(tphase)
    tphase = time_ns()
    linsys = _linsys(data, settings, work)
    linsys_time_sec = elapsed_time_sec(tphase)
    sol = Solution(T, data.n, data.m, data.p)
    sol.setup_time_sec = elapsed_time_sec(t0)
    sol.profile.problem_data_time_sec = problem_data_time_sec
    sol.profile.workspace_time_sec = workspace_time_sec
    sol.profile.linsys_time_sec = linsys_time_sec
    warmstart = Warmstart(T, data.n, data.m, data.p)
    return CoreSolver{T,Ti}(copy_settings(settings), data, scaling, work, linsys, sol, warmstart)
end

function _print_header(solver::CoreSolver{T}) where {T<:AbstractFloat}
    data = solver.data
    settings = solver.settings
    stats = _refresh_scaling_stats!(solver)
    io = settings.output
    @printf(io, "\n")
    @printf(io, "+-------------------------------------------------------+\n")
    @printf(io, "|     QOCO - Quadratic Objective Conic Optimizer        |\n")
    @printf(io, "|                    JuliaQOCO v%s                   |\n", string(pkgversion(@__MODULE__)))
    @printf(io, "+-------------------------------------------------------+\n")
    @printf(io, "| Problem Data:                                         |\n")
    @printf(io, "|     variables:        %-9d                       |\n", data.n)
    @printf(io, "|     constraints:      %-9d                       |\n", data.l + data.p + length(data.q))
    @printf(io, "|     eq constraints:   %-9d                       |\n", data.p)
    @printf(io, "|     ineq constraints: %-9d                       |\n", data.l)
    @printf(io, "|     soc constraints:  %-9d                       |\n", length(data.q))
    @printf(io, "|     nnz(P):           %-9d                       |\n", nnz(data.P) - length(data.Padded_idx))
    @printf(io, "|     nnz(A):           %-9d                       |\n", nnz(data.A))
    @printf(io, "|     nnz(G):           %-9d                       |\n", nnz(data.G))
    @printf(io, "| Scaling Statistics:                                   |\n")
    @printf(io, "|     Objective range      [%.0e, %.0e]               |\n", stats.obj_range_min, stats.obj_range_max)
    @printf(io, "|     Constraint range     [%.0e, %.0e]               |\n", stats.constraint_range_min, stats.constraint_range_max)
    @printf(io, "|     RHS range            [%.0e, %.0e]               |\n", stats.rhs_range_min, stats.rhs_range_max)
    @printf(io, "| Solver Settings:                                      |\n")
    algebra = "cached QDLDL"
    @printf(io, "|     algebra: %-27s              |\n", algebra)
    @printf(io, "|     max_iters: %-3d abstol: %3.2e reltol: %3.2e  |\n", settings.max_iters, settings.abstol, settings.reltol)
    @printf(io, "|     abstol_inacc: %3.2e reltol_inacc: %3.2e     |\n", settings.abstol_inacc, settings.reltol_inacc)
    @printf(io, "|     bisect_iters: %-2d iter_ref_iters: %-2d               |\n", settings.bisect_iters, settings.iter_ref_iters)
    @printf(io, "|     iter_ref_tol: %3.2e profile: %-5s               |\n", settings.iter_ref_tol, settings.profile ? "true" : "false")
    @printf(io, "|     ruiz_iters: %-2d kkt_static_reg: %3.2e           |\n", settings.ruiz_iters, settings.kkt_static_reg)
    @printf(io, "|     kkt_dynamic_reg: %3.2e                         |\n", settings.kkt_dynamic_reg)
    @printf(io, "+-------------------------------------------------------+\n")
    println(io)
    @printf(io, "+--------+-----------+------------+------------+------------+-----------+-----------+\n")
    @printf(io, "|  Iter  |   Pcost   |    Pres    |    Dres    |     Gap    |     Mu    |    Step   |\n")
    @printf(io, "+--------+-----------+------------+------------+------------+-----------+-----------+\n")
    return nothing
end

function _log_iter(solver::CoreSolver{T}) where {T<:AbstractFloat}
    io = solver.settings.output
    @printf(
        io,
        "|   %2d   | %+.2e | %+.3e | %+.3e | %+.3e | %+.2e |   %.3f   |\n",
        solver.solution.iters,
        solver.solution.obj,
        solver.solution.pres,
        solver.solution.dres,
        solver.solution.gap,
        solver.work.mu,
        solver.work.a,
    )
    @printf(io, "+--------+-----------+------------+------------+------------+-----------+-----------+\n")
    return nothing
end

function _print_footer(solver::CoreSolver{T}) where {T<:AbstractFloat}
    sol = solver.solution
    io = solver.settings.output
    @printf(io, "\n")
    @printf(io, "status:                %s\n", status_string(sol.status, sol.status_detail))
    @printf(io, "number of iterations:  %d\n", sol.iters)
    @printf(io, "result from iteration: %d\n", sol.result_iter)
    @printf(io, "result available:      %s\n", sol.result_available ? "yes" : "no")
    @printf(io, "objective:             %+.6e\n", sol.obj)
    @printf(io, "primal residual:       %.3e\n", sol.pres)
    @printf(io, "dual residual:         %.3e\n", sol.dres)
    @printf(io, "duality gap:           %.3e\n", sol.gap)
    @printf(io, "accuracy ratio:        %.3e\n", sol.quality)
    @printf(io, "setup time:            %.2e sec\n", sol.setup_time_sec)
    @printf(io, "solve time:            %.2e sec\n", sol.solve_time_sec)
    if solver.settings.profile
        profile = sol.profile
        @printf(io, "problem data time:     %.2e sec\n", profile.problem_data_time_sec)
        @printf(io, "workspace time:        %.2e sec\n", profile.workspace_time_sec)
        @printf(io, "linsys setup time:     %.2e sec\n", profile.linsys_time_sec)
        @printf(io, "initialize time:       %.2e sec\n", profile.initialize_time_sec)
        @printf(io, "residual/check time:   %.2e / %.2e sec\n", profile.residual_time_sec, profile.stopping_time_sec)
        @printf(io, "nt scale/update time:  %.2e / %.2e sec\n", profile.nt_scaling_time_sec, profile.nt_update_time_sec)
        @printf(io, "predictor time:        %.2e sec\n", profile.predictor_time_sec)
        @printf(io, "linsys solve/refine:   %.2e / %.2e sec\n", profile.linsys_solve_time_sec, profile.linsys_refine_time_sec)
        @printf(io, "linsys solves/refacs:  %d / %d\n", profile.linsys_solves, profile.nt_refactors)
        @printf(io, "factor retries/regularized pivots: %d / %d\n", profile.factorization_retries, profile.regularized_pivots)
        @printf(io, "warm start acc/rep/rej/retry: %d / %d / %d / %d\n",
            profile.warmstart_accepted, profile.warmstart_repaired,
            profile.warmstart_rejected, profile.warmstart_retries)
    end
    @printf(io, "\n")
    return nothing
end

function _reset_best_iterate!(solution::Solution{T}) where {T<:AbstractFloat}
    solution.best_metric = floatmax(T)
    solution.best_iter = 0
    solution.best_valid = false
    return solution
end

# Rank iterates by the shared normalized quality metric, which already scores
# a nonfinite or cone-invalid point as `Inf` and so can never select one.
# `assess_iterate!` must have been called for the current iterate.
function _record_best_iterate!(solver::CoreSolver{T}, iter::Int) where {T<:AbstractFloat}
    solution = solver.solution
    metric = solution.quality
    isfinite(metric) || return solution
    if !solution.best_valid || metric < solution.best_metric
        copyto!(solution.best_x, solver.work.x)
        copyto!(solution.best_s, solver.work.s)
        copyto!(solution.best_y, solver.work.y)
        copyto!(solution.best_z, solver.work.z)
        solution.best_metric = metric
        solution.best_iter = iter
        solution.best_valid = true
    end
    return solution
end

function _restore_best_iterate!(solver::CoreSolver)
    solver.solution.best_valid || return false
    copyto!(solver.work.x, solver.solution.best_x)
    copyto!(solver.work.s, solver.solution.best_s)
    copyto!(solver.work.y, solver.solution.best_y)
    copyto!(solver.work.z, solver.solution.best_z)
    return true
end

function _reset_dynamic_regularization!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    factor = solver.linsys.factor
    factor === nothing && return nothing
    factor.workspace.regularize_eps = solver.settings.kkt_dynamic_reg
    factor.workspace.regularize_delta = solver.settings.kkt_dynamic_reg
    return nothing
end

# The one place a result leaves the solver.
#
#  1. The most recently computed iterate has already been assessed by the
#     caller, including the final permitted step at the iteration limit.
#  2. The better of that iterate and the recorded best is selected using the
#     shared normalized quality metric.
#  3. Objective, residuals, complementarity and cone validity are recomputed
#     for whichever iterate was selected, so the reported metrics always
#     describe the vectors that are returned.
#  4. The selected iterate is unscaled and published.
#  5. A termination reason and result availability are assigned from the
#     recomputed metrics rather than from how the loop happened to exit.
#  6. The warm-start cache is updated only for a usable result.
#
# `allow_solved` is false when the solve ended in an exception or a stall, so
# that a recovered iterate can be reported as inaccurate at best.
function _finalize!(
    solver::CoreSolver{T},
    t0::UInt64,
    iters::Int,
    base_status::SolveStatus,
    base_detail::AbstractString,
    allow_solved::Bool,
) where {T<:AbstractFloat}
    settings = solver.settings
    sol = solver.solution
    sol.iters = iters

    current_metric = sol.quality
    if sol.best_valid && sol.best_metric < current_metric
        _restore_best_iterate!(solver)
        compute_kkt_residual!(solver)
        compute_mu!(solver)
        assess_iterate!(solver)
        sol.result_iter = sol.best_iter
    else
        sol.result_iter = iters
    end

    quality_inacc = solution_quality(solver, settings.abstol_inacc, settings.reltol_inacc)
    if allow_solved && sol.quality <= one(T)
        sol.status = QOCO_SOLVED
        sol.status_detail = ""
    elseif quality_inacc <= one(T)
        sol.status = QOCO_SOLVED_INACCURATE
        sol.status_detail = isempty(base_detail) ?
                            "met inaccurate tolerances" :
                            string(base_detail, "; met inaccurate tolerances")
    else
        sol.status = base_status
        sol.status_detail = base_detail
    end

    unscaled_solution!(sol, solver.data, solver.scaling, solver.work)
    sol.result_available =
        sol.cone_valid &&
        all_finite(sol.x) && all_finite(sol.s) &&
        all_finite(sol.y) && all_finite(sol.z) &&
        isfinite(sol.obj) && isfinite(sol.pres) && isfinite(sol.dres) && isfinite(sol.gap)
    sol.solve_time_sec = elapsed_time_sec(t0)
    _cache_solution_as_warmstart!(solver)
    settings.verbose && _print_footer(solver)
    return solver
end

# Exit taken when no iterate was ever computed, for example when the very
# first factorization or the initialization solve failed. Nothing is published:
# an all-zero or stale vector must not be presented as a newly computed
# solution.
function _finalize_without_result!(
    solver::CoreSolver{T},
    t0::UInt64,
    detail::AbstractString,
) where {T<:AbstractFloat}
    sol = solver.solution
    sol.iters = 0
    sol.result_iter = 0
    sol.status = QOCO_NUMERICAL_ERROR
    sol.status_detail = detail
    sol.result_available = false
    sol.cone_valid = false
    sol.quality = T(Inf)
    fill!(sol.x, zero(T))
    fill!(sol.s, zero(T))
    fill!(sol.y, zero(T))
    fill!(sol.z, zero(T))
    sol.obj = zero(T)
    sol.pres = T(Inf)
    sol.dres = T(Inf)
    sol.gap = T(Inf)
    clear_warmstart!(solver)
    sol.solve_time_sec = elapsed_time_sec(t0)
    solver.settings.verbose && _print_footer(solver)
    return solver
end

# The interior-point loop. `PROFILE` is a compile-time flag so that the timing
# instrumentation vanishes entirely from the fast path; the numerical work is
# written exactly once, which is what makes profiled and unprofiled runs
# bit-for-bit identical.
function _solve_loop!(
    solver::CoreSolver{T},
    t0::UInt64,
    iter_offset::Int,
    ::Val{PROFILE},
) where {T<:AbstractFloat,PROFILE}
    settings = solver.settings
    sol = solver.solution
    profile = sol.profile
    tphase = UInt64(0)
    iter = 0
    stopped = false
    # The budget is checked at iteration boundaries using a monotonic clock. A
    # single factorization call is not preemptible, so this bounds the number
    # of further iterations rather than guaranteeing a hard deadline.
    budget = settings.time_limit_sec
    out_of_time = false

    remaining = settings.max_iters - iter_offset
    while iter < remaining
        if isfinite(budget) && iter > 0 && elapsed_time_sec(t0) >= budget
            out_of_time = true
            break
        end
        PROFILE && (tphase = time_ns())
        compute_kkt_residual!(solver)
        PROFILE && (profile.residual_time_sec += elapsed_time_sec(tphase))
        PROFILE && (tphase = time_ns())
        compute_mu!(solver)
        PROFILE && (profile.mu_time_sec += elapsed_time_sec(tphase))
        PROFILE && (tphase = time_ns())
        stopped = check_stopping!(solver)
        _record_best_iterate!(solver, iter_offset + iter)
        PROFILE && (profile.stopping_time_sec += elapsed_time_sec(tphase))
        stopped && break

        PROFILE && (tphase = time_ns())
        compute_nt_scaling!(solver)
        PROFILE && (profile.nt_scaling_time_sec += elapsed_time_sec(tphase))
        PROFILE && (tphase = time_ns())
        update_nt_block!(solver)
        PROFILE && (profile.nt_update_time_sec += elapsed_time_sec(tphase))
        PROFILE && (tphase = time_ns())
        predictor_corrector!(solver)
        PROFILE && (profile.predictor_time_sec += elapsed_time_sec(tphase))

        iter += 1
        sol.iters = iter_offset + iter
        settings.verbose && _log_iter(solver)
    end

    if stopped
        allow_solved = sol.status != QOCO_NUMERICAL_ERROR
        return _finalize!(
            solver, t0, iter_offset + iter, sol.status, sol.status_detail, allow_solved,
        )
    end

    # Iteration or time limit. The step taken on the final permitted iteration
    # has not been looked at yet, so assess it before deciding anything.
    compute_kkt_residual!(solver)
    compute_mu!(solver)
    assess_iterate!(solver)
    _record_best_iterate!(solver, iter_offset + iter)
    status = out_of_time ? QOCO_TIME_LIMIT : QOCO_MAX_ITER
    detail = out_of_time ? "reached the solve time budget" : "reached iteration limit"
    return _finalize!(solver, t0, iter_offset + iter, status, detail, true)
end

function _solve!(solver::CoreSolver{T}) where {T<:AbstractFloat}
    validate_settings(solver.settings)
    sol = solver.solution
    sol.status = QOCO_UNSOLVED
    sol.status_detail = ""
    sol.iters = 0
    sol.result_iter = 0
    sol.result_available = false
    sol.cone_valid = false
    sol.quality = floatmax(T)
    reset_solve_profile!(sol.profile)
    _reset_best_iterate!(sol)
    _reset_dynamic_regularization!(solver)
    t0 = time_ns()
    solver.settings.verbose && _print_header(solver)
    if solver.data.n + solver.data.p + solver.data.m == 0
        sol.status = QOCO_SOLVED
        sol.status_detail = "empty problem"
        sol.obj = zero(T)
        sol.pres = zero(T)
        sol.dres = zero(T)
        sol.gap = zero(T)
        sol.quality = zero(T)
        sol.cone_valid = true
        sol.result_available = true
        sol.solve_time_sec = elapsed_time_sec(t0)
        _cache_solution_as_warmstart!(solver)
        solver.settings.verbose && _print_footer(solver)
        return solver
    end
    refresh_data_norms!(solver)
    _run_solve!(solver, t0, 0)
    if _should_retry_cold(solver)
        _retry_cold!(solver, t0)
    end
    return solver
end

# One attempt: initialize, iterate, finalize. Exceptions are turned into an
# honest status here rather than in the loop, so that both the warm attempt
# and any cold retry get the same treatment.
function _run_solve!(solver::CoreSolver{T}, t0::UInt64, iter_offset::Int) where {T<:AbstractFloat}
    sol = solver.solution
    initialized = false
    try
        if solver.settings.profile
            tphase = time_ns()
            initialize_ipm!(solver)
            sol.profile.initialize_time_sec += elapsed_time_sec(tphase)
        else
            initialize_ipm!(solver)
        end
        initialized = true
        return _solve_loop!(
            solver, t0, iter_offset, solver.settings.profile ? Val(true) : Val(false),
        )
    catch error
        error isa InterruptException && rethrow()
        detail = sprint(showerror, error)
        # Initialization failure leaves no iterate at all; anything later can
        # still fall back on a recorded best iterate.
        (initialized && sol.best_valid) ||
            return _finalize_without_result!(solver, t0, detail)
        return _finalize!(solver, t0, sol.iters, QOCO_NUMERICAL_ERROR, detail, false)
    end
end

# A reused start can pin the iterate to the wrong face of the cone, which
# shows up as an early stall rather than as slow progress. When that happens
# the solve is worth one, and only one, cold restart, sharing the same
# iteration and time budget as the attempt it replaces.
function _should_retry_cold(solver::CoreSolver{T}) where {T<:AbstractFloat}
    sol = solver.solution
    sol.profile.warmstart_accepted > 0 || return false
    sol.status in (QOCO_SOLVED, QOCO_SOLVED_INACCURATE, QOCO_TIME_LIMIT) && return false
    sol.iters < solver.settings.max_iters || return false
    if isfinite(solver.settings.time_limit_sec)
        sol.solve_time_sec < solver.settings.time_limit_sec || return false
    end
    return true
end

function _retry_cold!(solver::CoreSolver{T}, t0::UInt64) where {T<:AbstractFloat}
    sol = solver.solution
    completed = sol.iters
    # Seed the best-iterate slot with whatever the warm attempt produced, so
    # the shared selection metric in `_finalize!` keeps the better of the two
    # without needing a second set of buffers.
    if sol.result_available
        copyto!(sol.best_x, solver.work.x)
        copyto!(sol.best_s, solver.work.s)
        copyto!(sol.best_y, solver.work.y)
        copyto!(sol.best_z, solver.work.z)
        sol.best_metric = sol.quality
        sol.best_iter = sol.result_iter
        sol.best_valid = true
    else
        _reset_best_iterate!(sol)
    end
    clear_warmstart!(solver)
    _reset_dynamic_regularization!(solver)
    sol.profile.warmstart_retries += 1
    _run_solve!(solver, t0, completed)
    return solver
end

Base.summary(io::IO, solver::CoreSolver) = print(io, "JuliaQOCO solver ($(solver.data.n) vars, $(solver.data.p) eq, $(solver.data.m) cone rows)")
