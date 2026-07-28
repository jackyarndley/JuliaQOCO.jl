function _soc_offsets(l::Integer, qdims::AbstractVector{Ti}) where {Ti<:Integer}
    soc_offsets = Vector{Ti}(undef, length(qdims))
    Wfull_offsets = Vector{Ti}(undef, length(qdims))
    Wtri_offsets = Vector{Ti}(undef, length(qdims))
    soff = Ti(l + 1)
    foff = Ti(l + 1)
    toff = Ti(l + 1)
    @inbounds for i in eachindex(qdims)
        q = qdims[i]
        soc_offsets[i] = soff
        Wfull_offsets[i] = foff
        Wtri_offsets[i] = toff
        soff += q
        foff += q * q
        toff += q * (q + 1) ÷ 2
    end
    return soc_offsets, Wfull_offsets, Wtri_offsets
end

function _workspace(data::ProblemData{T,Ti}) where {T<:AbstractFloat,Ti<:Integer}
    Wnnz = kkt_nt_nnz(data.l, data.q)
    Wfullnnz = kkt_nt_full_nnz(data.l, data.q)
    maxq = isempty(data.q) ? 0 : maximum(data.q)
    soc_offsets, Wfull_offsets, Wtri_offsets = _soc_offsets(data.l, data.q)
    return Workspace{T,Ti}(
        zeros(T, data.n),
        zeros(T, data.m),
        zeros(T, data.p),
        zeros(T, data.m),
        zero(T),
        one(T),
        zero(T),
        zeros(T, Wnnz),
        zeros(T, Wfullnnz),
        zeros(T, Wfullnnz),
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
        Wfull_offsets,
        Wtri_offsets,
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
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
    n = Ti(length(c))
    A0 = A === nothing ? spzeros(T, 0, n) : copy(A)
    G0 = G === nothing ? spzeros(T, 0, n) : copy(G)
    b0 = b === nothing ? zeros(T, 0) : collect(b)
    h0 = h === nothing ? zeros(T, 0) : collect(h)
    P0 = P === nothing ? spzeros(T, n, n) : (istriu(P) ? copy(P) : SparseMatrixCSC(triu(P)))
    qv = Ti.(collect(q))

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
        Ti(l),
        qv,
        n,
        Ti(length(h0)),
        Ti(length(b0)),
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

function _refresh_scaling_stats!(solver::Solver)
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
        return LinearSystem{T,Ti,Nothing}(nothing, Ti[], Ti[], zeros(T, length(work.WtW)), Ti[], T[])
    end
    K, nt2kkt, ntdiag_positions, P2kkt, At2kkt, Gt2kkt = construct_kkt(data, settings, work)
    signs = vcat(ones(Int, data.n), -ones(Int, data.p + data.m))
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
    static2kkt = QDLDL.map_indices(factor, static2kkt)
    static_values = zeros(T, length(static2kkt))
    _fill_static_values!(static_values, data)
    return LinearSystem{T,Ti,typeof(factor)}(
        factor,
        nt2kkt,
        ntdiag_positions,
        zeros(T, length(work.WtW)),
        static2kkt,
        static_values,
    )
end

function Solver(
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
    work = _workspace(data)
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
    return Solver{T,Ti,typeof(linsys.factor)}(copy_settings(settings), data, scaling, work, linsys, sol, warmstart)
end

function _print_header(solver::Solver{T}) where {T<:AbstractFloat}
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
    @printf(io, "|     algebra: %-27s              |\n", "JuliaQOCO.QDLDL")
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

function _log_iter(solver::Solver{T}) where {T<:AbstractFloat}
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

function _print_footer(solver::Solver{T}) where {T<:AbstractFloat}
    sol = solver.solution
    io = solver.settings.output
    @printf(io, "\n")
    @printf(io, "status:                %s\n", status_string(sol.status, sol.status_detail))
    @printf(io, "number of iterations:  %d\n", sol.iters)
    @printf(io, "objective:             %+.6e\n", sol.obj)
    @printf(io, "primal residual:       %.3e\n", sol.pres)
    @printf(io, "dual residual:         %.3e\n", sol.dres)
    @printf(io, "duality gap:           %.3e\n", sol.gap)
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
    end
    @printf(io, "\n")
    return nothing
end

function _solve_fast_loop!(solver::Solver{T}, t0::UInt64) where {T<:AbstractFloat}
    for iter in 1:solver.settings.max_iters
        compute_kkt_residual!(solver)
        compute_objective!(solver)
        compute_mu!(solver)
        if check_stopping!(solver)
            solver.solution.solve_time_sec = elapsed_time_sec(t0)
            solver.solution.iters = iter - 1
            unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
            _cache_solution_as_warmstart!(solver)
            if solver.settings.verbose
                _print_footer(solver)
            end
            return solver
        end
        compute_nt_scaling!(solver)
        update_nt_block!(solver)
        predictor_corrector!(solver)
        solver.solution.iters = iter
        if solver.settings.verbose
            _log_iter(solver)
        end
    end

    solver.solution.status = QOCO_MAX_ITER
    solver.solution.status_detail = "reached iteration limit"
    solver.solution.solve_time_sec = elapsed_time_sec(t0)
    unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
    _cache_solution_as_warmstart!(solver)
    if solver.settings.verbose
        _print_footer(solver)
    end
    return solver
end

function _solve_profiled_loop!(solver::Solver{T}, t0::UInt64) where {T<:AbstractFloat}
    for iter in 1:solver.settings.max_iters
        tphase = time_ns()
        compute_kkt_residual!(solver)
        solver.solution.profile.residual_time_sec += elapsed_time_sec(tphase)
        tphase = time_ns()
        compute_objective!(solver)
        solver.solution.profile.objective_time_sec += elapsed_time_sec(tphase)
        tphase = time_ns()
        compute_mu!(solver)
        solver.solution.profile.mu_time_sec += elapsed_time_sec(tphase)
        tphase = time_ns()
        if check_stopping!(solver)
            solver.solution.profile.stopping_time_sec += elapsed_time_sec(tphase)
            solver.solution.solve_time_sec = elapsed_time_sec(t0)
            solver.solution.iters = iter - 1
            unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
            _cache_solution_as_warmstart!(solver)
            solver.settings.verbose && _print_footer(solver)
            return solver
        end
        solver.solution.profile.stopping_time_sec += elapsed_time_sec(tphase)
        tphase = time_ns()
        compute_nt_scaling!(solver)
        solver.solution.profile.nt_scaling_time_sec += elapsed_time_sec(tphase)
        tphase = time_ns()
        update_nt_block!(solver)
        solver.solution.profile.nt_update_time_sec += elapsed_time_sec(tphase)
        tphase = time_ns()
        predictor_corrector!(solver)
        solver.solution.profile.predictor_time_sec += elapsed_time_sec(tphase)
        solver.solution.iters = iter
        solver.settings.verbose && _log_iter(solver)
    end
    solver.solution.status = QOCO_MAX_ITER
    solver.solution.status_detail = "reached iteration limit"
    solver.solution.solve_time_sec = elapsed_time_sec(t0)
    unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
    _cache_solution_as_warmstart!(solver)
    solver.settings.verbose && _print_footer(solver)
    return solver
end

function solve!(solver::Solver{T}) where {T<:AbstractFloat}
    validate_settings(solver.settings)
    solver.solution.status = QOCO_UNSOLVED
    solver.solution.status_detail = ""
    solver.solution.iters = 0
    reset_solve_profile!(solver.solution.profile)
    t0 = time_ns()
    solver.settings.verbose && _print_header(solver)
    if solver.data.n + solver.data.p + solver.data.m == 0
        solver.solution.status = QOCO_SOLVED
        solver.solution.status_detail = "empty problem"
        solver.solution.obj = zero(T)
        solver.solution.pres = zero(T)
        solver.solution.dres = zero(T)
        solver.solution.gap = zero(T)
        solver.solution.solve_time_sec = elapsed_time_sec(t0)
        _cache_solution_as_warmstart!(solver)
        solver.settings.verbose && _print_footer(solver)
        return solver
    end
    if solver.settings.profile
        tphase = time_ns()
        initialize_ipm!(solver)
        solver.solution.profile.initialize_time_sec += elapsed_time_sec(tphase)
        return _solve_profiled_loop!(solver, t0)
    end
    initialize_ipm!(solver)
    return _solve_fast_loop!(solver, t0)
end

function solve(
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
    solver = Solver(P, c, A, b, G, h, l, q; settings = settings)
    solve!(solver)
    return solver
end

function solve(
    P::Union{Nothing,SparseMatrixCSC{T,Int}},
    c::AbstractVector{T};
    A::Union{Nothing,SparseMatrixCSC{T,Int}} = nothing,
    b::Union{Nothing,AbstractVector{T}} = nothing,
    G::Union{Nothing,SparseMatrixCSC{T,Int}} = nothing,
    h::Union{Nothing,AbstractVector{T}} = nothing,
    l::Integer = 0,
    q::AbstractVector{<:Integer} = Int[],
    settings::Settings{T} = default_settings(T),
) where {T<:AbstractFloat}
    return solve(P, c, A, b, G, h, l, q; settings = settings)
end

Base.summary(io::IO, solver::Solver) = print(io, "JuliaQOCO solver ($(solver.data.n) vars, $(solver.data.p) eq, $(solver.data.m) cone rows)")
