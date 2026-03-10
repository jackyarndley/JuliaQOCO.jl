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
    nnz_est = nnz(U) + (n - count_diag_upper(U))
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Vector{Ti}()
    nzval = Vector{T}()
    sizehint!(rowval, nnz_est)
    sizehint!(nzval, nnz_est)
    colptr[1] = one(Ti)
    @inbounds for j in 1:n
        diag_found = false
        for k in U.colptr[j]:(U.colptr[j + 1] - 1)
            row = U.rowval[k]
            val = U.nzval[k]
            if row == j
                val += reg
                diag_found = true
            end
            push!(rowval, row)
            push!(nzval, val)
        end
        if !diag_found
            push!(rowval, Ti(j))
            push!(nzval, reg)
        end
        colptr[j + 1] = Ti(length(rowval) + 1)
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
    return SparseMatrixCSC(transpose(A))
end

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
