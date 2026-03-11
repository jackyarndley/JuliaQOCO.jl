function validate_data(
    P::Union{Nothing,SparseMatrixCSC},
    c::AbstractVector,
    A::Union{Nothing,SparseMatrixCSC},
    b::Union{Nothing,AbstractVector},
    G::Union{Nothing,SparseMatrixCSC},
    h::Union{Nothing,AbstractVector},
    l::Integer,
    q::AbstractVector{<:Integer},
)
    q === nothing && throw(ArgumentError("q must be provided"))
    l >= 0 || throw(ArgumentError("l must be nonnegative"))
    all(>=(0), q) || throw(ArgumentError("SOC dimensions must be nonnegative"))
    (A === nothing) == (b === nothing) || throw(ArgumentError("A and b must either both be provided or both be omitted"))
    (G === nothing) == (h === nothing) || throw(ArgumentError("G and h must either both be provided or both be omitted"))
    P === nothing || size(P, 1) == size(P, 2) || throw(ArgumentError("P must be square"))
    n = length(c)
    P === nothing || size(P, 2) == n || throw(ArgumentError("P column dimension must match length(c)"))
    A === nothing || size(A, 2) == n || throw(ArgumentError("A column dimension must match length(c)"))
    G === nothing || size(G, 2) == n || throw(ArgumentError("G column dimension must match length(c)"))
    if A !== nothing
        size(A, 1) == length(b) || throw(ArgumentError("size(A,1) must match length(b)"))
    end
    if G !== nothing
        size(G, 1) == length(h) || throw(ArgumentError("size(G,1) must match length(h)"))
        l + sum(q) == size(G, 1) || throw(ArgumentError("l + sum(q) must equal size(G,1)"))
    else
        l + sum(q) == 0 || throw(ArgumentError("If G is omitted, l + sum(q) must be zero"))
    end
    return nothing
end

function initialize_scaling(data::ProblemData{T}) where {T<:AbstractFloat}
    return Scaling{T}(
        zeros(T, data.n + data.p + data.m),
        ones(T, data.n),
        ones(T, data.p),
        ones(T, data.m),
        ones(T, data.n),
        ones(T, data.p),
        ones(T, data.m),
        zeros(T, data.n),
        zeros(T, data.n),
        one(T),
        one(T),
    )
end

function ruiz_equilibration!(data::ProblemData{T,Ti}, scaling::Scaling{T}, ruiz_iters::Int) where {T<:AbstractFloat,Ti<:Integer}
    fill!(scaling.Druiz, one(T))
    fill!(scaling.Eruiz, one(T))
    fill!(scaling.Fruiz, one(T))
    fill!(scaling.Dinvruiz, one(T))
    fill!(scaling.Einvruiz, one(T))
    fill!(scaling.Finvruiz, one(T))
    scaling.k = one(T)
    scaling.kinv = one(T)

    D = @view scaling.delta[1:data.n]
    E = @view scaling.delta[(data.n + 1):(data.n + data.p)]
    F = @view scaling.delta[(data.n + data.p + 1):(data.n + data.p + data.m)]
    Anorm = scaling.Anorm
    Gnorm = scaling.Gnorm

    for _ in 1:ruiz_iters
        fill!(D, zero(T))
        g = inf_norm(data.c)
        pinf_mean = zero(T)
        if nnz(data.P) > 0
            col_inf_norm_upper_symmetric!(D, data.P)
            @inbounds for j in eachindex(D)
                pinf_mean += D[j]
            end
            pinf_mean /= max(one(T), T(data.n))
        end
        g = max(g, pinf_mean)
        g = safe_div(one(T), g)
        scaling.k *= g

        if nnz(data.A) > 0
            col_inf_norm_matrix!(Anorm, data.A)
            @inbounds for j in eachindex(D)
                D[j] = max(D[j], Anorm[j])
            end
        end
        if nnz(data.G) > 0
            col_inf_norm_matrix!(Gnorm, data.G)
            @inbounds for j in eachindex(D)
                D[j] = max(D[j], Gnorm[j])
            end
        end
        @inbounds for j in eachindex(D)
            D[j] = safe_div(one(T), sqrt(D[j]))
        end

        if data.p > 0
            col_inf_norm_matrix!(E, data.At)
            @inbounds for k in eachindex(E)
                E[k] = safe_div(one(T), sqrt(E[k]))
            end
        end

        if data.m > 0
            col_inf_norm_matrix!(F, data.Gt)
            @inbounds for k in eachindex(F)
                F[k] = safe_div(one(T), sqrt(F[k]))
            end
            idx = data.l + 1
            for qk in data.q
                @inbounds for t in (idx + 1):(idx + qk - 1)
                    F[t] = F[idx]
                end
                idx += qk
            end
        end

        if nnz(data.P) > 0
            scale!(data.P.nzval, g)
            row_col_scale_matrix!(data.P, D, D)
        end

        scale!(data.c, g)
        ew_product!(data.c, data.c, D)

        row_col_scale_matrix!(data.A, E, D)
        row_col_scale_matrix!(data.G, F, D)
        row_col_scale_matrix!(data.At, D, E)
        row_col_scale_matrix!(data.Gt, D, F)

        ew_product!(scaling.Druiz, scaling.Druiz, D)
        ew_product!(scaling.Eruiz, scaling.Eruiz, E)
        ew_product!(scaling.Fruiz, scaling.Fruiz, F)
    end

    ew_product!(data.b, data.b, scaling.Eruiz)
    ew_product!(data.h, data.h, scaling.Fruiz)

    reciprocal!(scaling.Dinvruiz, scaling.Druiz)
    reciprocal!(scaling.Einvruiz, scaling.Eruiz)
    reciprocal!(scaling.Finvruiz, scaling.Fruiz)
    scaling.kinv = safe_div(one(T), scaling.k)
    data.stats = compute_scaling_statistics(data)
    data.stats_dirty = false
    return nothing
end

function unscaled_solution!(
    solution::Solution{T},
    data::ProblemData{T},
    scaling::Scaling{T},
    work::Workspace{T},
) where {T<:AbstractFloat}
    @inbounds for i in eachindex(solution.x, work.x, scaling.Druiz)
        solution.x[i] = work.x[i] * scaling.Druiz[i]
    end
    @inbounds for i in eachindex(solution.s, work.s, scaling.Finvruiz)
        solution.s[i] = work.s[i] * scaling.Finvruiz[i]
    end
    @inbounds for i in eachindex(solution.y, work.y, scaling.Eruiz)
        solution.y[i] = work.y[i] * scaling.Eruiz[i] * scaling.kinv
    end
    @inbounds for i in eachindex(solution.z, work.z, scaling.Fruiz)
        solution.z[i] = work.z[i] * scaling.Fruiz[i] * scaling.kinv
    end
    return solution
end
