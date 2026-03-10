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

    data = ProblemData{T,Ti}(
        P0,
        collect(c),
        A0,
        create_transposed_matrix(A0),
        b0,
        G0,
        create_transposed_matrix(G0),
        h0,
        Ti(l),
        qv,
        n,
        Ti(length(h0)),
        Ti(length(b0)),
        ScalingStats(T),
    )
    scaling = initialize_scaling(data)
    ruiz_equilibration!(data, scaling, settings.ruiz_iters)
    data.P = regularize_P(data.P, settings.kkt_static_reg)
    data.stats = compute_scaling_statistics(data)
    return data, scaling
end

function _linsys(data::ProblemData{T,Ti}, settings::Settings{T}, work::Workspace{T,Ti}) where {T<:AbstractFloat,Ti<:Integer}
    K, nt2kkt, ntdiag_positions = construct_kkt(data, settings, work)
    signs = vcat(ones(Int, data.n), -ones(Int, data.p + data.m))
    factor = QDLDL.qdldl(
        K;
        Dsigns = signs,
        regularize_eps = settings.kkt_dynamic_reg,
        regularize_delta = settings.kkt_dynamic_reg,
    )
    return LinearSystem{T,Ti,typeof(factor)}(factor, nt2kkt, ntdiag_positions, zeros(T, length(work.WtW)))
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
    return Solver{T,Ti,typeof(linsys.factor)}(copy_settings(settings), data, scaling, work, linsys, sol)
end

function solve!(solver::Solver{T}) where {T<:AbstractFloat}
    validate_settings(solver.settings)
    solver.solution.status = QOCO_UNSOLVED
    solver.solution.iters = 0
    t0 = time()
    initialize_ipm!(solver)

    for iter in 1:solver.settings.max_iters
        compute_kkt_residual!(solver)
        compute_objective!(solver)
        compute_mu!(solver)

        if check_stopping!(solver)
            solver.solution.solve_time_sec = time() - t0
            solver.solution.iters = iter - 1
            unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
            return solver
        end

        compute_nt_scaling!(solver)
        update_nt_block!(solver)
        predictor_corrector!(solver)
        solver.solution.iters = iter
    end

    solver.solution.status = QOCO_MAX_ITER
    solver.solution.solve_time_sec = time() - t0
    unscaled_solution!(solver.solution, solver.data, solver.scaling, solver.work)
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
