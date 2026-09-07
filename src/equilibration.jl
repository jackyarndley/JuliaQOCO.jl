# Scaling conventions used throughout the solver.
#
# With positive diagonal matrices D (variables), E (equality rows), F (cone
# rows) and a positive objective scale k, the internally stored data is
#
#     P_hat = k D P D  (before the static regularization shift is added)
#     c_hat = k D c
#     A_hat = E A D,   b_hat = E b
#     G_hat = F G D,   h_hat = F h
#
# and the iterates relate to original units by
#
#     x = D x_hat,   s = F^-1 s_hat,   y = E y_hat / k,   z = F z_hat / k.
#
# The identities that follow, and that the stopping criteria rely on, are
#
#     s'z      = dot(s_hat, z_hat) / k
#     x'Px     = dot(x_hat, P_hat x_hat) / k
#     c'x      = dot(c_hat, x_hat) / k
#     |c|_inf  = |D^-1 c_hat|_inf / k
#     |s|_inf  = |F^-1 s_hat|_inf
#
# `scaling.Druiz` stores D, `scaling.Dinvruiz` stores D^-1, and likewise for
# E and F. `scaling.k` stores k and `scaling.kinv` stores 1/k.

# Largest objective scale the equilibration is allowed to introduce. Without a
# bound, a problem whose objective data is tiny (or exactly zero, as in a pure
# feasibility problem) drives k to infinity and destroys every unscaling.
const MAX_OBJECTIVE_SCALE = 1e12

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
    function validate_sparse(M, name)
        M === nothing && return nothing
        M.colptr[1] == 1 || throw(ArgumentError("$name CSC column pointers must start at one"))
        M.colptr[end] == nnz(M) + 1 || throw(ArgumentError("$name CSC column pointers are inconsistent"))
        all(isfinite, M.nzval) || throw(ArgumentError("$name must contain only finite values"))
        @inbounds for col in 1:size(M, 2)
            previous = 0
            for position in M.colptr[col]:(M.colptr[col + 1] - 1)
                row = M.rowval[position]
                1 <= row <= size(M, 1) || throw(ArgumentError("$name row index is out of bounds"))
                row > previous || throw(ArgumentError("$name contains duplicate or unsorted row indices"))
                previous = row
            end
        end
        return nothing
    end
    all(isfinite, c) || throw(ArgumentError("c must contain only finite values"))
    A !== nothing && all(isfinite, b) || A === nothing || throw(ArgumentError("b must contain only finite values"))
    G !== nothing && all(isfinite, h) || G === nothing || throw(ArgumentError("h must contain only finite values"))
    validate_sparse(P, "P")
    validate_sparse(A, "A")
    validate_sparse(G, "G")
    q === nothing && throw(ArgumentError("q must be provided"))
    l >= 0 || throw(ArgumentError("l must be nonnegative"))
    # A zero-dimensional second-order cone has no head entry, which every cone
    # kernel assumes exists. Reject it here rather than letting the kernels
    # read past the block.
    all(qi -> qi >= 2, q) || throw(ArgumentError("SOC dimensions must be at least two"))
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

@inline function _ruiz_inverse_sqrt(value::T) where {T<:AbstractFloat}
    (isfinite(value) && value > safe_div_eps(T)) || return one(T)
    scale = inv(sqrt(value))
    return isfinite(scale) && scale > zero(T) ? scale : one(T)
end

# Objective scale for one Ruiz sweep. A zero or unusable objective norm means
# there is nothing to equilibrate, so the neutral scale one is returned rather
# than a huge sentinel, which would overflow k. The result is also bounded so
# that a merely tiny objective cannot run k away over the sweeps.
@inline function _objective_scale(norm_value::T, k_so_far::T) where {T<:AbstractFloat}
    (isfinite(norm_value) && norm_value > safe_div_eps(T)) || return one(T)
    g = inv(norm_value)
    isfinite(g) && g > zero(T) || return one(T)
    limit = T(MAX_OBJECTIVE_SCALE)
    product = k_so_far * g
    if product > limit
        g = limit / k_so_far
    elseif product < inv(limit)
        g = inv(limit) / k_so_far
    end
    return isfinite(g) && g > zero(T) ? g : one(T)
end

# Cone rows must share one common scale so that the row scaling maps the
# second-order cone onto itself. The head row alone is a poor representative
# when the tail rows are much larger or smaller, so the geometric mean of the
# per-row Ruiz scales is used instead. Logs keep the mean well behaved for
# large cones.
function _apply_common_soc_scales!(F::AbstractVector{T}, l::Integer, qdims::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    idx = Int(l) + 1
    for qk in qdims
        log_sum = zero(T)
        @inbounds for t in idx:(idx + qk - 1)
            log_sum += log(F[t])
        end
        common = exp(log_sum / T(qk))
        isfinite(common) && common > zero(T) || (common = one(T))
        @inbounds for t in idx:(idx + qk - 1)
            F[t] = common
        end
        idx += qk
    end
    return F
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
        g = _objective_scale(g, scaling.k)
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
            D[j] = _ruiz_inverse_sqrt(D[j])
        end

        if data.p > 0
            col_inf_norm_matrix!(E, data.At)
            @inbounds for k in eachindex(E)
                E[k] = _ruiz_inverse_sqrt(E[k])
            end
        end

        if data.m > 0
            col_inf_norm_matrix!(F, data.Gt)
            @inbounds for k in eachindex(F)
                F[k] = _ruiz_inverse_sqrt(F[k])
            end
            _apply_common_soc_scales!(F, data.l, data.q)
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
    isfinite(scaling.k) && scaling.k > zero(T) || (scaling.k = one(T))
    scaling.kinv = positive_inv(scaling.k)
    data.stats = compute_scaling_statistics(data)
    data.stats_dirty = false
    return nothing
end

# Convert the internal scaled iterate held in `work` into the original units of
# the user problem. This is the only place the four coordinate conversions are
# written down for output; `_warmstart_to_original!` and
# `_warmstart_to_scaled!` in updates.jl are their inverses.
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
