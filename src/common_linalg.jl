function count_diag_upper(P::SparseMatrixCSC)
    count = 0
    @inbounds for j in 1:size(P, 2)
        start = P.colptr[j]
        stop = P.colptr[j + 1] - 1
        if start <= stop && P.rowval[stop] == j
            count += 1
        end
    end
    return count
end

function regularize_P(P::SparseMatrixCSC{T,Ti}, reg::T) where {T<:AbstractFloat,Ti<:Integer}
    U = istriu(P) ? copy(P) : SparseMatrixCSC(triu(P))
    n = size(U, 1)
    diag_count = count_diag_upper(U)
    missing_diag = n - diag_count
    nnz_est = nnz(U) + missing_diag
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Vector{Ti}(undef, nnz_est)
    nzval = Vector{T}(undef, nnz_est)
    colptr[1] = one(Ti)
    nz = 1
    @inbounds for j in 1:n
        diag_found = false
        for k in U.colptr[j]:(U.colptr[j + 1] - 1)
            row = U.rowval[k]
            val = U.nzval[k]
            if row == j
                val += reg
                diag_found = true
            end
            rowval[nz] = row
            nzval[nz] = val
            nz += 1
        end
        if !diag_found
            rowval[nz] = Ti(j)
            nzval[nz] = reg
            nz += 1
        end
        colptr[j + 1] = Ti(nz)
    end
    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end

function construct_identity_upper(::Type{T}, n::Integer, λ::T) where {T<:AbstractFloat}
    Ti = Int
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Vector{Ti}(undef, n)
    nzval = Vector{T}(undef, n)
    @inbounds for j in 1:n
        colptr[j] = j
        rowval[j] = j
        nzval[j] = λ
    end
    colptr[n + 1] = n + 1
    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end

function create_transposed_matrix(A::SparseMatrixCSC{T,Ti}) where {T,Ti<:Integer}
    m, n = size(A)
    nnzA = nnz(A)
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
    rowval = Vector{Ti}(undef, nnzA)
    nzval = Vector{T}(undef, nnzA)
    @inbounds for j in 1:n
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            row = A.rowval[k]
            dest = nextptr[row]
            rowval[dest] = Ti(j)
            nzval[dest] = A.nzval[k]
            nextptr[row] += one(Ti)
        end
    end
    return SparseMatrixCSC(n, m, colptr, rowval, nzval)
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

regularize_existing_P!(P::SparseMatrixCSC{T,Ti}, reg::T) where {T<:AbstractFloat,Ti<:Integer} = shift_diag!(P, reg)
unregularize_P!(P::SparseMatrixCSC{T,Ti}, reg::T) where {T<:AbstractFloat,Ti<:Integer} = shift_diag!(P, -reg)

function row_col_scale_matrix!(A::SparseMatrixCSC{T,Ti}, row_scale::AbstractVector{T}, col_scale::AbstractVector{T}) where {T<:AbstractFloat,Ti<:Integer}
    @inbounds for j in 1:size(A, 2)
        α = col_scale[j]
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            A.nzval[k] *= α * row_scale[A.rowval[k]]
        end
    end
    return A
end

function col_inf_norm_matrix!(dest::AbstractVector{T}, A::SparseMatrixCSC{T}) where {T<:AbstractFloat}
    fill!(dest, zero(T))
    @inbounds for j in 1:size(A, 2)
        nrm = zero(T)
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            nrm = max(nrm, abs(A.nzval[k]))
        end
        dest[j] = nrm
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
            if i != j
                dest[i] = max(dest[i], v)
            end
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
            v = P.nzval[k]
            y[i] += v * xj
            if i != j
                y[j] += v * x[i]
            end
        end
    end
    return y
end
