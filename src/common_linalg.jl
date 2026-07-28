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
    return first(regularize_P_with_info(P, reg))
end

function regularize_P_with_info(P::SparseMatrixCSC{T,Ti}, reg::T) where {T<:AbstractFloat,Ti<:Integer}
    U = istriu(P) ? copy(P) : SparseMatrixCSC(triu(P))
    n = size(U, 1)
    nnz_est = nnz(U) + (n - count_diag_upper(U))
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Vector{Ti}()
    nzval = Vector{T}()
    padded_idx = Vector{Ti}()
    sizehint!(rowval, nnz_est)
    sizehint!(nzval, nnz_est)
    sizehint!(padded_idx, n - count_diag_upper(U))
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
            push!(padded_idx, Ti(length(rowval)))
        end
        colptr[j + 1] = Ti(length(rowval) + 1)
    end
    return SparseMatrixCSC(n, n, colptr, rowval, nzval), padded_idx
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
    return first(create_transposed_matrix_with_map(A))
end

function create_transposed_matrix_with_map(A::SparseMatrixCSC{T,Ti}) where {T,Ti<:Integer}
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
    to_original = Vector{Ti}(undef, nnzA)
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
    original_to_transposed = Vector{Ti}(undef, length(transposed_to_original))
    @inbounds for transposed_index in eachindex(transposed_to_original)
        original_to_transposed[transposed_to_original[transposed_index]] = Ti(transposed_index)
    end
    return original_to_transposed
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

regularize_existing_P!(P::SparseMatrixCSC{T,Ti}, reg::T) where {T<:AbstractFloat,Ti<:Integer} = shift_diag!(P, reg)
unregularize_P!(P::SparseMatrixCSC{T,Ti}, reg::T) where {T<:AbstractFloat,Ti<:Integer} = shift_diag!(P, -reg)

function copy_original_P_values!(
    P::SparseMatrixCSC{T,Ti},
    Pxnew::AbstractVector{T},
    padded_idx::AbstractVector{Ti},
) where {T<:AbstractFloat,Ti<:Integer}
    nnz(P) - length(padded_idx) == length(Pxnew) || throw(ArgumentError("Pxnew must match the original nnz(P) before regularization"))
    padded_pos = 1
    next_padded = isempty(padded_idx) ? typemax(Ti) : padded_idx[padded_pos]
    src = 1
    @inbounds for k in 1:nnz(P)
        if k == next_padded
            padded_pos += 1
            next_padded = padded_pos <= length(padded_idx) ? padded_idx[padded_pos] : typemax(Ti)
        else
            P.nzval[k] = Pxnew[src]
            src += 1
        end
    end
    return P
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

@generated function generated_mul_upper_symmetric!(
    y::AbstractVector{T},
    values::AbstractVector{T},
    x::AbstractVector{T},
    ::QDLDL.GeneratedPattern{
        AP,
        AI,
        LP,
        LI,
        ROWPTR,
        ROWCOLS,
        ROWVALS,
        HAS_SIGNS,
        QDLDL.GeneratedSparseOps{PCP,PRI,ACP,ARI,GCP,GRI},
    },
) where {
    T,
    AP,
    AI,
    LP,
    LI,
    ROWPTR,
    ROWCOLS,
    ROWVALS,
    HAS_SIGNS,
    PCP,
    PRI,
    ACP,
    ARI,
    GCP,
    GRI,
}
    body = Expr(:block)
    push!(body.args, :(fill!(y, zero(T))))
    for column in 1:(length(PCP) - 1)
        xcolumn = gensym(:xcolumn)
        push!(body.args, :($xcolumn = x[$column]))
        for position in PCP[column]:(PCP[column + 1] - 1)
            row = PRI[position]
            push!(
                body.args,
                :(y[$row] = muladd(values[$position], $xcolumn, y[$row])),
            )
            if row != column
                push!(
                    body.args,
                    :(y[$column] = muladd(values[$position], x[$row], y[$column])),
                )
            end
        end
    end
    push!(body.args, :(return y))
    return quote
        @inbounds begin
            $body
        end
    end
end

@generated function generated_mul_A!(
    y::AbstractVector{T},
    values::AbstractVector{T},
    x::AbstractVector{T},
    ::QDLDL.GeneratedPattern{
        AP,
        AI,
        LP,
        LI,
        ROWPTR,
        ROWCOLS,
        ROWVALS,
        HAS_SIGNS,
        QDLDL.GeneratedSparseOps{PCP,PRI,ACP,ARI,GCP,GRI},
    },
) where {
    T,
    AP,
    AI,
    LP,
    LI,
    ROWPTR,
    ROWCOLS,
    ROWVALS,
    HAS_SIGNS,
    PCP,
    PRI,
    ACP,
    ARI,
    GCP,
    GRI,
}
    body = Expr(:block)
    push!(body.args, :(fill!(y, zero(T))))
    for column in 1:(length(ACP) - 1)
        xcolumn = gensym(:xcolumn)
        push!(body.args, :($xcolumn = x[$column]))
        for position in ACP[column]:(ACP[column + 1] - 1)
            row = ARI[position]
            push!(
                body.args,
                :(y[$row] = muladd(values[$position], $xcolumn, y[$row])),
            )
        end
    end
    push!(body.args, :(return y))
    return quote
        @inbounds begin
            $body
        end
    end
end

@generated function generated_mul_At!(
    y::AbstractVector{T},
    values::AbstractVector{T},
    x::AbstractVector{T},
    ::QDLDL.GeneratedPattern{
        AP,
        AI,
        LP,
        LI,
        ROWPTR,
        ROWCOLS,
        ROWVALS,
        HAS_SIGNS,
        QDLDL.GeneratedSparseOps{PCP,PRI,ACP,ARI,GCP,GRI},
    },
) where {
    T,
    AP,
    AI,
    LP,
    LI,
    ROWPTR,
    ROWCOLS,
    ROWVALS,
    HAS_SIGNS,
    PCP,
    PRI,
    ACP,
    ARI,
    GCP,
    GRI,
}
    body = Expr(:block)
    for column in 1:(length(ACP) - 1)
        acc = gensym(:acc)
        push!(body.args, :($acc = zero(T)))
        for position in ACP[column]:(ACP[column + 1] - 1)
            row = ARI[position]
            push!(
                body.args,
                :($acc = muladd(values[$position], x[$row], $acc)),
            )
        end
        push!(body.args, :(y[$column] = $acc))
    end
    push!(body.args, :(return y))
    return quote
        @inbounds begin
            $body
        end
    end
end

@generated function generated_mul_G!(
    y::AbstractVector{T},
    values::AbstractVector{T},
    x::AbstractVector{T},
    ::QDLDL.GeneratedPattern{
        AP,
        AI,
        LP,
        LI,
        ROWPTR,
        ROWCOLS,
        ROWVALS,
        HAS_SIGNS,
        QDLDL.GeneratedSparseOps{PCP,PRI,ACP,ARI,GCP,GRI},
    },
) where {
    T,
    AP,
    AI,
    LP,
    LI,
    ROWPTR,
    ROWCOLS,
    ROWVALS,
    HAS_SIGNS,
    PCP,
    PRI,
    ACP,
    ARI,
    GCP,
    GRI,
}
    body = Expr(:block)
    push!(body.args, :(fill!(y, zero(T))))
    for column in 1:(length(GCP) - 1)
        xcolumn = gensym(:xcolumn)
        push!(body.args, :($xcolumn = x[$column]))
        for position in GCP[column]:(GCP[column + 1] - 1)
            row = GRI[position]
            push!(
                body.args,
                :(y[$row] = muladd(values[$position], $xcolumn, y[$row])),
            )
        end
    end
    push!(body.args, :(return y))
    return quote
        @inbounds begin
            $body
        end
    end
end

@generated function generated_mul_Gt!(
    y::AbstractVector{T},
    values::AbstractVector{T},
    x::AbstractVector{T},
    ::QDLDL.GeneratedPattern{
        AP,
        AI,
        LP,
        LI,
        ROWPTR,
        ROWCOLS,
        ROWVALS,
        HAS_SIGNS,
        QDLDL.GeneratedSparseOps{PCP,PRI,ACP,ARI,GCP,GRI},
    },
) where {
    T,
    AP,
    AI,
    LP,
    LI,
    ROWPTR,
    ROWCOLS,
    ROWVALS,
    HAS_SIGNS,
    PCP,
    PRI,
    ACP,
    ARI,
    GCP,
    GRI,
}
    body = Expr(:block)
    for column in 1:(length(GCP) - 1)
        acc = gensym(:acc)
        push!(body.args, :($acc = zero(T)))
        for position in GCP[column]:(GCP[column + 1] - 1)
            row = GRI[position]
            push!(
                body.args,
                :($acc = muladd(values[$position], x[$row], $acc)),
            )
        end
        push!(body.args, :(y[$column] = $acc))
    end
    push!(body.args, :(return y))
    return quote
        @inbounds begin
            $body
        end
    end
end
