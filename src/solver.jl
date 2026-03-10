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
    P1, Padded_idx = regularize_P_with_info(P0, zero(T))
    data = ProblemData{T,Ti}(
        P1,
        collect(c),
        A0,
        At0,
        AtoAt,
        b0,
        G0,
        Gt0,
        GtoGt,
        h0,
        Ti(l),
        qv,
        n,
        Ti(length(h0)),
        Ti(length(b0)),
        Padded_idx,
        ScalingStats(T),
    )
    scaling = initialize_scaling(data)
    ruiz_equilibration!(data, scaling, settings.ruiz_iters)
    regularize_existing_P!(data.P, settings.kkt_static_reg)
    data.stats = compute_scaling_statistics(data)
    return data, scaling
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
    t0 = time()
    validate_settings(settings)
    data, scaling = _problem_data(P, c, A, b, G, h, l, q, settings)
    work = _workspace(data)
    linsys = _linsys(data, settings, work)
    sol = Solution(T, data.n, data.m, data.p)
    sol.setup_time_sec = time() - t0
    warmstart = Warmstart(T, data.n, data.m, data.p)
    return Solver{T,Ti,typeof(linsys.factor)}(copy_settings(settings), data, scaling, work, linsys, sol, warmstart)
end

function _print_header(solver::Solver{T}) where {T<:AbstractFloat}
    data = solver.data
    settings = solver.settings
    stats = data.stats
    @printf("\n")
    @printf("+-------------------------------------------------------+\n")
    @printf("|     QOCO - Quadratic Objective Conic Optimizer        |\n")
    @printf("|                    JuliaQOCO v%s                    |\n", string(pkgversion(@__MODULE__)))
    @printf("+-------------------------------------------------------+\n")
    @printf("| Problem Data:                                         |\n")
    @printf("|     variables:        %-9d                       |\n", data.n)
    @printf("|     constraints:      %-9d                       |\n", data.l + data.p + length(data.q))
    @printf("|     eq constraints:   %-9d                       |\n", data.p)
    @printf("|     ineq constraints: %-9d                       |\n", data.l)
    @printf("|     soc constraints:  %-9d                       |\n", length(data.q))
    @printf("|     nnz(P):           %-9d                       |\n", nnz(data.P))
    @printf("|     nnz(A):           %-9d                       |\n", nnz(data.A))
    @printf("|     nnz(G):           %-9d                       |\n", nnz(data.G))
    @printf("| Scaling Statistics:                                   |\n")
    @printf("|     Objective range      [%.0e, %.0e]               |\n", stats.obj_range_min, stats.obj_range_max)
    @printf("|     Constraint range     [%.0e, %.0e]               |\n", stats.constraint_range_min, stats.constraint_range_max)
    @printf("|     RHS range            [%.0e, %.0e]               |\n", stats.rhs_range_min, stats.rhs_range_max)
    @printf("| Solver Settings:                                      |\n")
    @printf("|     algebra: %-27s              |\n", "QDLDL.jl")
    @printf("|     max_iters: %-3d abstol: %3.2e reltol: %3.2e  |\n", settings.max_iters, settings.abstol, settings.reltol)
    @printf("|     abstol_inacc: %3.2e reltol_inacc: %3.2e     |\n", settings.abstol_inacc, settings.reltol_inacc)
    @printf("|     bisect_iters: %-2d iter_ref_iters: %-2d               |\n", settings.bisect_iters, settings.iter_ref_iters)
    @printf("|     ruiz_iters: %-2d kkt_static_reg: %3.2e           |\n", settings.ruiz_iters, settings.kkt_static_reg)
    @printf("|     kkt_dynamic_reg: %3.2e                         |\n", settings.kkt_dynamic_reg)
    @printf("+-------------------------------------------------------+\n")
    println()
    @printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n")
    @printf("|  Iter  |   Pcost   |    Pres    |    Dres    |     Gap    |     Mu    |    Step   |\n")
    @printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n")
    return nothing
end

function _log_iter(solver::Solver{T}) where {T<:AbstractFloat}
    @printf(
        "|   %2d   | %+.2e | %+.3e | %+.3e | %+.3e | %+.2e |   %.3f   |\n",
        solver.solution.iters,
        solver.solution.obj,
        solver.solution.pres,
        solver.solution.dres,
        solver.solution.gap,
        solver.work.mu,
        solver.work.a,
    )
    @printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n")
    return nothing
end

function _print_footer(solver::Solver{T}) where {T<:AbstractFloat}
    sol = solver.solution
    @printf("\n")
    @printf("status:                %s\n", status_string(sol.status, sol.status_detail))
    @printf("number of iterations:  %d\n", sol.iters)
    @printf("objective:             %+.6e\n", sol.obj)
    @printf("primal residual:       %.3e\n", sol.pres)
    @printf("dual residual:         %.3e\n", sol.dres)
    @printf("duality gap:           %.3e\n", sol.gap)
    @printf("setup time:            %.2e sec\n", sol.setup_time_sec)
    @printf("solve time:            %.2e sec\n", sol.solve_time_sec)
    @printf("\n")
    return nothing
end

function solve!(solver::Solver{T}) where {T<:AbstractFloat}
    validate_settings(solver.settings)
    solver.solution.status = QOCO_UNSOLVED
    solver.solution.status_detail = ""
    solver.solution.iters = 0
    t0 = time()
    if solver.settings.verbose
        _print_header(solver)
    end
    if solver.data.n + solver.data.p + solver.data.m == 0
        solver.solution.status = QOCO_SOLVED
        solver.solution.status_detail = "empty problem"
        solver.solution.obj = zero(T)
        solver.solution.pres = zero(T)
        solver.solution.dres = zero(T)
        solver.solution.gap = zero(T)
        solver.solution.solve_time_sec = time() - t0
        _cache_solution_as_warmstart!(solver)
        if solver.settings.verbose
            _print_footer(solver)
        end
        return solver
    end
    initialize_ipm!(solver)

    for iter in 1:solver.settings.max_iters
        compute_kkt_residual!(solver)
        compute_objective!(solver)
        compute_mu!(solver)

        if check_stopping!(solver)
            solver.solution.solve_time_sec = time() - t0
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
    solver.solution.solve_time_sec = time() - t0
    unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
    _cache_solution_as_warmstart!(solver)
    if solver.settings.verbose
        _print_footer(solver)
    end
    return solver
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
