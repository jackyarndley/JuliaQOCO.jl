# Convexity validation for the quadratic objective.
#
# The stored Hessian is `P_hat = k D P D` plus a static regularization shift on
# the diagonal. Both k > 0 and the diagonal D > 0, so `P_hat` without the shift
# is a congruence transform of P and has the same inertia: verifying the stored
# matrix therefore verifies the user matrix, and no unscaled copy is needed.
#
# Three tests are used, in increasing cost, and each is conclusive where it
# applies:
#
#  1. A diagonal pattern is positive semidefinite exactly when every diagonal
#     entry is nonnegative. O(nnz), and this is the common case for the
#     quadratic penalties that appear in sequential convex programming.
#  2. Below `dense_limit`, a dense symmetric eigenvalue decomposition.
#  3. Above it, a Cholesky factorization of `P + shift*I`. Success proves
#     `P + shift*I` is positive definite and hence that the smallest eigenvalue
#     of P exceeds `-shift`, which is exactly the semidefinite tolerance the
#     dense test applies. This is a genuine positive-definiteness test of a
#     shifted matrix, not an inertia count read off a sign-forced regularized
#     LDL factorization, which would prove nothing about P.

function is_diagonal_pattern(P::SparseMatrixCSC)
    @inbounds for j in 1:size(P, 2)
        for k in P.colptr[j]:(P.colptr[j + 1] - 1)
            P.rowval[k] == j || return false
        end
    end
    return true
end

# Smallest tolerated eigenvalue, relative to the magnitude of the matrix.
function _psd_tolerance(P::SparseMatrixCSC{T}, static_reg::T) where {T<:AbstractFloat}
    scale = max(one(T), inf_norm(P.nzval))
    return max(sqrt(eps(T)) * scale, T(2) * static_reg)
end

# `static_reg` is the shift that `regularize_existing_P!` added to the stored
# diagonal, and is removed here so that the mathematical Hessian is tested.
function is_positive_semidefinite(
    P::SparseMatrixCSC{T},
    static_reg::T,
    dense_limit::Integer,
) where {T<:AbstractFloat}
    nnz(P) > 0 || return true
    all(isfinite, P.nzval) || return false
    tolerance = _psd_tolerance(P, static_reg)
    n = size(P, 1)
    if is_diagonal_pattern(P)
        @inbounds for k in 1:nnz(P)
            P.nzval[k] - static_reg >= -tolerance || return false
        end
        return true
    end
    if n <= dense_limit
        dense = Matrix(Symmetric(P))
        @inbounds for i in 1:n
            dense[i, i] -= static_reg
        end
        return minimum(eigvals(Symmetric(dense))) >= -tolerance
    end
    return _shifted_cholesky_succeeds(P, tolerance - static_reg)
end

# Cholesky of the stored upper-triangular Hessian plus `shift` on the
# diagonal. The sparse factorization is used where the element type supports
# it, since a large sequential-convex-programming Hessian is sparse and a
# dense O(n^3) decomposition on every objective replacement is exactly the
# cost this policy exists to avoid.
function _shifted_cholesky_succeeds(P::SparseMatrixCSC{Float64}, shift::Float64)
    factor = cholesky(Symmetric(P, :U); shift = shift, check = false)
    return issuccess(factor)
end

function _shifted_cholesky_succeeds(P::SparseMatrixCSC{T}, shift::T) where {T<:AbstractFloat}
    dense = Matrix(Symmetric(P))
    @inbounds for i in 1:size(dense, 1)
        dense[i, i] += shift
    end
    return isposdef(Symmetric(dense))
end

function regularize_P_with_info(P::SparseMatrixCSC{T,Ti}, reg::T) where {T<:AbstractFloat,Ti<:Integer}
    U = istriu(P) ? copy(P) : SparseMatrixCSC(triu(P))
    n = size(U, 1)
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Ti[]
    nzval = T[]
    padded_idx = Ti[]
    sizehint!(rowval, nnz(U) + n)
    sizehint!(nzval, nnz(U) + n)
    colptr[1] = one(Ti)
    @inbounds for j in 1:n
        diag_found = false
        for k in U.colptr[j]:(U.colptr[j + 1] - 1)
            row = U.rowval[k]
            push!(rowval, row)
            push!(nzval, row == j ? U.nzval[k] + reg : U.nzval[k])
            diag_found |= row == j
        end
        if !diag_found
            push!(rowval, Ti(j))
            push!(nzval, reg)
            push!(padded_idx, Ti(length(rowval)))
        end
        colptr[j + 1] = Ti(length(rowval) + 1)
    end
    return SparseMatrixCSC(n, n, colptr, rowval, nzval), padded_idx
end

function create_transposed_matrix_with_map(A::SparseMatrixCSC{T,Ti}) where {T,Ti<:Integer}
    m, n = size(A)
    counts = zeros(Ti, m)
    @inbounds for row in A.rowval
        counts[row] += one(Ti)
    end
    colptr = Vector{Ti}(undef, m + 1)
    colptr[1] = one(Ti)
    @inbounds for j in 1:m
        colptr[j + 1] = colptr[j] + counts[j]
    end
    nextptr = copy(colptr)
    rowval = Vector{Ti}(undef, nnz(A))
    nzval = Vector{T}(undef, nnz(A))
    to_original = Vector{Ti}(undef, nnz(A))
    @inbounds for j in 1:n
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            row = A.rowval[k]
            dest = nextptr[row]
            rowval[dest] = Ti(j)
            nzval[dest] = A.nzval[k]
            to_original[dest] = Ti(k)
            nextptr[row] += one(Ti)
        end
    end
    return SparseMatrixCSC(n, m, colptr, rowval, nzval), to_original
end

function inverse_entry_map(transposed_to_original::AbstractVector{Ti}) where {Ti<:Integer}
    result = Vector{Ti}(undef, length(transposed_to_original))
    @inbounds for i in eachindex(transposed_to_original)
        result[transposed_to_original[i]] = Ti(i)
    end
    return result
end

function entry_columns(A::SparseMatrixCSC{T,Ti}) where {T,Ti<:Integer}
    columns = Vector{Ti}(undef, nnz(A))
    @inbounds for col in 1:size(A, 2)
        for position in A.colptr[col]:(A.colptr[col + 1] - 1)
            columns[position] = Ti(col)
        end
    end
    return columns
end

function shift_diag!(P::SparseMatrixCSC{T,Ti}, reg::T) where {T<:AbstractFloat,Ti<:Integer}
    @inbounds for j in 1:size(P, 2)
        for k in P.colptr[j]:(P.colptr[j + 1] - 1)
            if P.rowval[k] == j
                P.nzval[k] += reg
                break
            end
        end
    end
    return P
end

regularize_existing_P!(P::SparseMatrixCSC, reg) = shift_diag!(P, reg)
unregularize_P!(P::SparseMatrixCSC, reg) = shift_diag!(P, -reg)

function copy_original_P_values!(P::SparseMatrixCSC{T,Ti}, values::AbstractVector{T}, padded_idx::AbstractVector{Ti}) where {T<:AbstractFloat,Ti<:Integer}
    length(values) == nnz(P) - length(padded_idx) || throw(ArgumentError("values must match the original nnz(P)"))
    padded_pos = 1
    next_padded = isempty(padded_idx) ? typemax(Ti) : padded_idx[1]
    source = 1
    @inbounds for position in eachindex(P.nzval)
        if position == next_padded
            padded_pos += 1
            next_padded = padded_pos <= length(padded_idx) ? padded_idx[padded_pos] : typemax(Ti)
        else
            P.nzval[position] = values[source]
            source += 1
        end
    end
    return P
end

function row_col_scale_matrix!(A::SparseMatrixCSC{T,Ti}, row_scale::AbstractVector{T}, col_scale::AbstractVector{T}) where {T<:AbstractFloat,Ti<:Integer}
    @inbounds for j in 1:size(A, 2)
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            A.nzval[k] *= col_scale[j] * row_scale[A.rowval[k]]
        end
    end
    return A
end

function col_inf_norm_matrix!(dest::AbstractVector{T}, A::SparseMatrixCSC{T}) where {T<:AbstractFloat}
    fill!(dest, zero(T))
    @inbounds for j in 1:size(A, 2)
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            dest[j] = max(dest[j], abs(A.nzval[k]))
        end
    end
    return dest
end

function col_inf_norm_upper_symmetric!(dest::AbstractVector{T}, P::SparseMatrixCSC{T}) where {T<:AbstractFloat}
    fill!(dest, zero(T))
    @inbounds for j in 1:size(P, 2)
        for k in P.colptr[j]:(P.colptr[j + 1] - 1)
            i = P.rowval[k]
            v = abs(P.nzval[k])
            dest[j] = max(dest[j], v)
            i != j && (dest[i] = max(dest[i], v))
        end
    end
    return dest
end

function mul_upper_symmetric!(y::AbstractVector{T}, P::SparseMatrixCSC{T,Ti}, x::AbstractVector{T}) where {T<:AbstractFloat,Ti<:Integer}
    fill!(y, zero(T))
    @inbounds for j in 1:size(P, 2)
        xj = x[j]
        for k in P.colptr[j]:(P.colptr[j + 1] - 1)
            i = P.rowval[k]
            value = P.nzval[k]
            y[i] = muladd(value, xj, y[i])
            i != j && (y[j] = muladd(value, x[i], y[j]))
        end
    end
    return y
end
