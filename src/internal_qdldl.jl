# Vendored and modified from oxfordcontrol/QDLDL.jl (Apache-2.0),
# commit 9f5ed76cf88a117bd124a0c723d2fb992b766ba0.
#
# This copy is maintained inside JuliaQOCO so the factorization and update
# path can be specialized for this solver and optimized for lower allocations.
# See `licenses\\QDLDL-LICENSE` for the upstream license text.
module InternalQDLDL

export qdldl, \, solve, solve!, refactor!, update_values!, scale_values!,  positive_inertia, regularized_entries

using AMD, SparseArrays
using LinearAlgebra: istriu, triu, Diagonal

const QDLDL_UNKNOWN = -1;
const QDLDL_USED   = true;
const QDLDL_UNUSED = false;


struct QDLDLWorkspace{Tf<:AbstractFloat,Ti<:Integer,TA<:Union{Vector{Ti},Nothing},TS<:Union{Vector{Ti},Nothing}}

    #internal workspace data
    etree::Vector{Ti}
    Lnz::Vector{Ti}
    yidx::Vector{Ti}
    elim_buffer::Vector{Ti}
    lnext::Vector{Ti}
    bwork::Vector{Bool}
    factor_fwork::Vector{Tf}
    solve_fwork::Vector{Tf}

    #L matrix row indices and data
    Ln::Int         #always Int since SparseMatrixCSC does it this way
    Lp::Vector{Ti}
    Li::Vector{Ti}
    Lx::Vector{Tf}

    #D and its inverse
    D::Vector{Tf}
    Dinv::Vector{Tf}

    #cached row-wise symbolic pattern for fast numeric refactorization
    rowptr::Vector{Ti}
    rowcols::Vector{Ti}
    rowvals::Vector{Ti}
    pattern_initialized::Ref{Bool}

    #number of positive values in D
    positive_inertia::Ref{Ti}

    #The upper triangular matrix factorisation target
    #This is the post ordering PAPt of the original data
    triuA::SparseMatrixCSC{Tf,Ti}

    #mapping from entries in the triu form
    #of the original input to the post ordering
    #triu form used for the factorization
    #this can be used when modifying entries
    #of the data matrix for refactoring
    AtoPAPt::TA

    #regularization parameters
    Dsigns::TS
    regularize_eps::Tf
    regularize_delta::Tf

    #number of regularized entries in D
    #length 1 vector instead of ref to avoid allocations
    #while maintaining immutability
    regularize_count::Vector{Ti}

end

function QDLDLWorkspace(triuA::SparseMatrixCSC{Tf,Ti},
                        AtoPAPt::TA,
                        Dsigns::TS,
                        regularize_eps::Tf,
                        regularize_delta::Tf
) where {Tf<:AbstractFloat,Ti<:Integer,TA<:Union{Vector{Ti},Nothing},TS<:Union{Vector{Ti},Nothing}}

    etree  = Vector{Ti}(undef,triuA.n)
    Lnz    = Vector{Ti}(undef,triuA.n)
    yidx   = Vector{Ti}(undef,triuA.n)
    elim_buffer = Vector{Ti}(undef,triuA.n)
    lnext  = Vector{Ti}(undef,triuA.n)
    bwork  = Vector{Bool}(undef,triuA.n)
    factor_fwork = zeros(Tf, triuA.n)
    solve_fwork = Vector{Tf}(undef, triuA.n)

    #compute elimination tree using QDLDL converted code
    sumLnz = QDLDL_etree!(triuA.n,triuA.colptr,triuA.rowval,yidx,Lnz,etree)

    if(sumLnz < 0)
        error("Input matrix is not upper triangular or has an empty column")
    end

    #allocate space for the L matrix row indices and data
    Ln = triuA.n
    Lp = Vector{Ti}(undef,triuA.n + 1)
    Li = Vector{Ti}(undef,sumLnz)
    Lx = Vector{Tf}(undef,sumLnz)

    #allocate for D and D inverse
    D  = Vector{Tf}(undef,triuA.n)
    Dinv = Vector{Tf}(undef,triuA.n)

    #allocate cached row-wise symbolic pattern storage
    rowptr = Vector{Ti}(undef, triuA.n + 1)
    rowcols = Vector{Ti}(undef, sumLnz)
    rowvals = Vector{Ti}(undef, sumLnz)
    pattern_initialized = Ref{Bool}(false)

    #allocate for positive inertia count.  -1 to
    #start since we haven't counted anything yet
    positive_inertia = Ref{Ti}(-1)

    #number of regularized entries in D. None to start
    regularize_count = zeros(Ti,1)

    QDLDLWorkspace(etree,Lnz,yidx,elim_buffer,lnext,bwork,factor_fwork,solve_fwork,
                   Ln,Lp,Li,Lx,D,Dinv,rowptr,rowcols,rowvals,pattern_initialized,positive_inertia,triuA,
                   AtoPAPt, Dsigns,regularize_eps,
                   regularize_delta,regularize_count)

end

struct QDLDLFactorisation{
    Tf<:AbstractFloat,
    Ti<:Integer,
    TP<:Union{Nothing,Vector{Ti}},
    TIP<:Union{Nothing,Vector{Ti}},
    TW<:QDLDLWorkspace{Tf,Ti},
}

    #permutation vector (nothing if no permutation)
    perm::TP
    #inverse permutation (nothing if no permutation)
    iperm::TIP
    #lower triangular factor
    L::SparseMatrixCSC{Tf,Ti}
    #Inverse of D matrix in ldl
    Dinv::Diagonal{Tf,Vector{Tf}}
    #workspace data
    workspace::TW
    #is it logical factorisation only?
    logical::Ref{Bool}
end




# Usage :
# qdldl(A) uses the default AMD ordering
# qdldl(A,perm = p) uses a caller specified ordering
# qdldl(A,perm = nothing) factors without reordering
#
# qdldl(A,logical = true) produces a logical factorisation only
#
# qdldl(A,signs = s, thresh_eps = ϵ, thresh_delta = δ) produces
# a factorization with dynamic regularization based on the vector
# of signs in s and using regularization parameters (ϵ,δ).  The
# scalars (ϵ,δ) = (1e-12,1e-7) by default.   By default s = nothing,
# and no regularization is performed.
#
# qdldl(A,amd_dense_scale = s) scales AMD.AMD_DENSE by a factor s :
# (s = 1.0 by default).   This is only used if no perm parameter 
# is provided. 

function qdldl(A::SparseMatrixCSC{Tf,Ti};
               amd_dense_scale::Tf = Tf(1.0),
               perm::Union{Array{Ti},Nothing}=_get_amd_ordering(A,amd_dense_scale),
               logical::Bool=false,
               Dsigns::Union{Array{Ti},Nothing} = nothing,
               regularize_eps::Tf = Tf(1e-12),
               regularize_delta::Tf = Tf(1e-7),
              ) where {Tf<:AbstractFloat, Ti<:Integer}

    #store the inverse permutation to enable matrix updates
    iperm = perm === nothing ? nothing : invperm(perm)

    if(!istriu(A))
        A = triu(A)
    else
        #either way, we take an internal copy
        A = deepcopy(A)
    end


    #permute using symperm, producing a triu matrix to factor
    if perm !== nothing
        A, AtoPAPt = permute_symmetric(A, iperm)  #returns an upper triangular matrix
    else
        AtoPAPt = nothing
    end

    #hold an internal copy of the (possibly permuted)
    #vector of signs if one was specified
    if(Dsigns !== nothing)
        mysigns = similar(Dsigns)
        if(perm === nothing)
            mysigns .= Dsigns
        else
            permute!(mysigns,Dsigns,perm)
        end
    else
        mysigns = nothing
    end

    #allocate workspace
    workspace = QDLDLWorkspace(A,AtoPAPt,mysigns,regularize_eps,regularize_delta)

    #factor the matrix
    factor!(workspace,logical)

    #make user-friendly factors
    L = SparseMatrixCSC(workspace.Ln,
                        workspace.Ln,
                        workspace.Lp,
                        workspace.Li,
                        workspace.Lx)
    Dinv = Diagonal(workspace.Dinv)

    #Psss a Ref{Bool} to the constructor since QDLDLFactorisation
    #is immutable.   All internal functions will just use a Bool

    return QDLDLFactorisation(perm, iperm, L, Dinv, workspace, Ref{Bool}(logical))

end

function positive_inertia(F::QDLDLFactorisation)
    F.workspace.positive_inertia[]
end

function regularized_entries(F::QDLDLFactorisation)
    F.workspace.regularize_count[1]
end

@inline _mapped_index(::Nothing, index::Ti) where {Ti<:Integer} = index
@inline _mapped_index(AtoPAPt::AbstractVector{Ti}, index::Ti) where {Ti<:Integer} = AtoPAPt[index]

function map_indices(F::QDLDLFactorisation{Tf,Ti}, indices::AbstractVector{Ti}) where {Tf<:AbstractFloat,Ti<:Integer}
    mapped = similar(indices)
    AtoPAPt = F.workspace.AtoPAPt
    if AtoPAPt === nothing
        copyto!(mapped, indices)
    else
        @inbounds @simd for i in eachindex(indices)
            mapped[i] = AtoPAPt[indices[i]]
        end
    end
    return mapped
end

@inline function update_values!(F::QDLDLFactorisation{Tf,Ti}, index::Ti, value::Tf) where {Tf<:Real,Ti<:Integer}
    F.workspace.triuA.nzval[_mapped_index(F.workspace.AtoPAPt, index)] = value
    return nothing
end

function update_values!(
    F::QDLDLFactorisation{Tf,Ti},
    indices::AbstractVector{Ti},
    values::AbstractVector{Tf},
) where {Tf<:Real,Ti<:Integer}
    @boundscheck length(indices) == length(values) || throw(DimensionMismatch("indices and values must have matching lengths"))
    nzval = F.workspace.triuA.nzval
    AtoPAPt = F.workspace.AtoPAPt
    if AtoPAPt === nothing
        @inbounds @simd for i in eachindex(indices, values)
            nzval[indices[i]] = values[i]
        end
    else
        @inbounds @simd for i in eachindex(indices, values)
            nzval[AtoPAPt[indices[i]]] = values[i]
        end
    end
    return nothing
end

function update_values_internal!(
    F::QDLDLFactorisation{Tf,Ti},
    indices::AbstractVector{Ti},
    values::AbstractVector{Tf},
) where {Tf<:Real,Ti<:Integer}
    @boundscheck length(indices) == length(values) || throw(DimensionMismatch("indices and values must have matching lengths"))
    nzval = F.workspace.triuA.nzval
    @inbounds @simd for i in eachindex(indices, values)
        nzval[indices[i]] = values[i]
    end
    return nothing
end

@inline function scale_values!(F::QDLDLFactorisation{Tf,Ti}, index::Ti, scale::Tf) where {Tf<:Real,Ti<:Integer}
    F.workspace.triuA.nzval[_mapped_index(F.workspace.AtoPAPt, index)] *= scale
    return nothing
end

function scale_values!(
    F::QDLDLFactorisation{Tf,Ti},
    indices::AbstractVector{Ti},
    scale::Tf,
) where {Tf<:Real,Ti<:Integer}
    nzval = F.workspace.triuA.nzval
    AtoPAPt = F.workspace.AtoPAPt
    if AtoPAPt === nothing
        @inbounds @simd for i in eachindex(indices)
            nzval[indices[i]] *= scale
        end
    else
        @inbounds @simd for i in eachindex(indices)
            nzval[AtoPAPt[indices[i]]] *= scale
        end
    end
    return nothing
end


function Base.:\(F::QDLDLFactorisation,b)
    return solve(F,b)
end


function refactor!(F::QDLDLFactorisation)

    #It never makes sense to call refactor for a logical
    #factorization since it will always be the same.  Calling
    #this function implies that we want a numerical factorization

    F.logical[] = false  #in case not already

    factor!(F.workspace,F.logical[])
end


function factor!(workspace::QDLDLWorkspace{Tf,Ti},logical::Bool) where {Tf<:AbstractFloat,Ti<:Integer}

    if(logical)
        workspace.Lx   .= 1
        workspace.D    .= 1
        workspace.Dinv .= 1
    end

    A = workspace.triuA
    posDCount = if logical || !workspace.pattern_initialized[]
        QDLDL_factor!(A.n,A.colptr,A.rowval,A.nzval,
                      workspace.Lp,
                      workspace.Li,
                      workspace.Lx,
                      workspace.D,
                      workspace.Dinv,
                      workspace.Lnz,
                      workspace.etree,
                      workspace.bwork,
                      workspace.yidx,
                      workspace.elim_buffer,
                      workspace.lnext,
                      workspace.factor_fwork,
                      logical,
                      workspace.Dsigns,
                      workspace.regularize_eps,
                      workspace.regularize_delta,
                      workspace.regularize_count,
                      workspace.rowptr,
                      workspace.rowcols,
                      workspace.rowvals,
                      !workspace.pattern_initialized[])
    else
        QDLDL_numeric_factor!(A.n,
                             A.colptr,
                             A.rowval,
                             A.nzval,
                             workspace.Lp,
                             workspace.Li,
                             workspace.Lx,
                             workspace.D,
                             workspace.Dinv,
                             workspace.rowptr,
                             workspace.rowcols,
                             workspace.rowvals,
                             workspace.factor_fwork,
                             workspace.Dsigns,
                             workspace.regularize_eps,
                             workspace.regularize_delta,
                             workspace.regularize_count)
    end

    if(posDCount < 0)
        error("Zero entry in D (matrix is not quasidefinite)")
    end

    workspace.pattern_initialized[] = true

    workspace.positive_inertia[] = posDCount

    return nothing

end


# Solves Ax = b using LDL factors for A.
# Returns x, preserving b
function solve(F::QDLDLFactorisation,b)
    x = copy(b)
    solve!(F,x)
    return x
end

# Solves Ax = b using LDL factors for A.
# Solves in place (x replaces b)
function solve!(F::QDLDLFactorisation,b)

    #bomb if logical factorisation only
    if F.logical[]
        error("Can't solve with logical factorisation only")
    end

    perm = F.perm
    iperm = F.iperm

    #permute b
    tmp = perm === nothing ? b : permute!(F.workspace.solve_fwork,b,perm)

    QDLDL_solve!(F.workspace.Ln,
                 F.workspace.Lp,
                 F.workspace.Li,
                 F.workspace.Lx,
                 F.workspace.Dinv,
                 tmp)

    #inverse permutation
    if perm !== nothing
        inverse_permute!(b, F.workspace.solve_fwork, iperm)
    end

    return nothing
end



# Compute the elimination tree for a quasidefinite matrix
# in compressed sparse column form.

function QDLDL_etree!(n,Ap,Ai,work,Lnz,etree)

    i = 1
    @inbounds while i <= n
        # zero out Lnz and work.  Set all etree values to unknown
        work[i]  = 0
        Lnz[i]   = 0
        etree[i] = QDLDL_UNKNOWN

        #Abort if A doesn't have at least one entry
        #one entry in every column
        if(Ap[i] == Ap[i+1])
            return -1
        end
        i += 1
    end

    j = 1
    @inbounds while j <= n
        work[j] = j
        p = Ap[j]
        pstop = Ap[j + 1] - 1
        while p <= pstop
            i = Ai[p]
            if(i > j)
                return -1
            end
            @inbounds while(work[i] != j)
                if(etree[i] == QDLDL_UNKNOWN)
                    etree[i] = j
                end
                Lnz[i] += 1        #nonzeros in this column
                work[i] = j
                i = etree[i]
            end
            p += 1
        end #end while p
        j += 1
    end

    #tally the total nonzeros
    sumLnz = sum(Lnz)

    return sumLnz
end




function QDLDL_factor!(
        n,
        Ap,
        Ai,
        Ax,
        Lp,
        Li,
        Lx,
        D,
        Dinv,
        Lnz,
        etree,
        bwork,
        yIdx,
        elimBuffer,
        LNextSpaceInCol,
        fwork,
        logicalFactor::Bool,
        Dsigns,
        regularize_eps,
        regularize_delta,
        regularize_count,
        rowptr,
        rowcols,
        rowvals,
        record_pattern::Bool,
)

    positiveValuesInD  = 0
    regularize_count[1] = 0

    yMarkers        = bwork
    yVals           = fwork
    zeroT = zero(eltype(D))
    rowwrite = one(eltype(Lp))

    Lp[1] = 1 #first column starts at index one / Julia is 1 indexed
    if record_pattern
        rowptr[1] = one(eltype(rowptr))
        rowptr[2] = one(eltype(rowptr))
    end

    i = 1
    @inbounds while i <= n

        #compute L column indices
        Lp[i+1] = Lp[i] + Lnz[i]   #cumsum, total at the end

        # set all Yidx to be 'unused' initially
        #in each column of L, the next available space
        #to start is just the first space in the column
        yMarkers[i]  = QDLDL_UNUSED
        yVals[i]     = zeroT
        D[i]         = zeroT
        LNextSpaceInCol[i] = Lp[i]
        i += 1
    end

    if(!logicalFactor)
        # First element of the diagonal D.
        D[1]     = Ax[1]
        if(Dsigns !== nothing && Dsigns[1]*D[1] < regularize_eps)
            D[1] = regularize_delta * Dsigns[1]
            regularize_count[1] += 1
        end

        if(D[1] == zeroT) return -1 end
        if(D[1]  > zeroT) positiveValuesInD += 1 end
        Dinv[1] = 1/D[1];
    end

    #Start from second row here. The upper LH corner is trivially 0
    #in L b/c we are only computing the subdiagonal elements
    k = 2
    @inbounds while k <= n

        #NB : For each k, we compute a solution to
        #y = L(0:(k-1),0:k-1))\b, where b is the kth
        #column of A that sits above the diagonal.
        #The solution y is then the kth row of L,
        #with an implied '1' at the diagonal entry.

        #number of nonzeros in this row of L
        nnzY = 0  #number of elements in this row

        #This loop determines where nonzeros
        #will go in the kth row of L, but doesn't
        #compute the actual values
        i = Ap[k]
        istop = Ap[k + 1] - 1
        while i <= istop

            bidx = Ai[i]   # we are working on this element of b

            #Initialize D[k] as the element of this column
            #corresponding to the diagonal place.  Don't use
            #this element as part of the elimination step
            #that computes the k^th row of L
            if(bidx == k)
                D[k] = Ax[i];
                i += 1
                continue
            end

            yVals[bidx] = Ax[i]   # initialise y(bidx) = b(bidx)

            # use the forward elimination tree to figure
            # out which elements must be eliminated after
            # this element of b
            nextIdx = bidx

            if(yMarkers[nextIdx] == QDLDL_UNUSED)  #this y term not already visited

                yMarkers[nextIdx] = QDLDL_USED     #I touched this one
                elimBuffer[1]     = nextIdx  # It goes at the start of the current list
                nnzE              = 1         #length of unvisited elimination path from here

                nextIdx = etree[bidx];

                @inbounds while(nextIdx != QDLDL_UNKNOWN && nextIdx < k)
                    if(yMarkers[nextIdx] == QDLDL_USED) break; end

                    yMarkers[nextIdx] = QDLDL_USED;   #I touched this one
                    #NB: Julia is 1-indexed, so I increment nnzE first here,
                    #not after writing into elimBuffer as in the C version
                    nnzE += 1                   #the list is one longer than before
                    elimBuffer[nnzE] = nextIdx; #It goes in the current list
                    nextIdx = etree[nextIdx];   #one step further along tree

                end #end while

                # now I put the buffered elimination list into
                # my current ordering in reverse order
                @inbounds while(nnzE != 0)
                    #NB: inc/dec reordered relative to C because
                    #the arrays are 1 indexed
                    nnzY += 1;
                    yIdx[nnzY] = elimBuffer[nnzE];
                    nnzE -= 1;
                end #end while
            end #end if

            i += 1
        end #end while i

        #This for loop places nonzeros values in the k^th row
        i = nnzY
        while i >= 1

            #which column are we working on?
            cidx = yIdx[i]

            # loop along the elements in this
            # column of L and subtract to solve to y
            tmpIdx = LNextSpaceInCol[cidx];

            #don't compute Lx for logical factorisation
            #this is not implemented in the C version
            if(!logicalFactor)
                yVals_cidx = yVals[cidx]
                j = Lp[cidx]
                jstop = tmpIdx - 1
                while j <= jstop
                    rowj = Li[j]
                    yVals[rowj] = muladd(-Lx[j], yVals_cidx, yVals[rowj])
                    j += 1
                end

                #Now I have the cidx^th element of y = L\b.
                #so compute the corresponding element of
                #this row of L and put it into the right place
                Lx[tmpIdx] = yVals_cidx *Dinv[cidx]

                #D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
                D[k] = muladd(-yVals_cidx, Lx[tmpIdx], D[k])
            end

            #also record which row it went into
            Li[tmpIdx] = k
            if record_pattern
                rowcols[rowwrite] = cidx
                rowvals[rowwrite] = tmpIdx
                rowwrite += 1
            end

            LNextSpaceInCol[cidx] += 1

            #reset the yvalues and indices back to zero and QDLDL_UNUSED
            #once I'm done with them
            yVals[cidx]     = zeroT
            yMarkers[cidx]  = QDLDL_UNUSED

            i -= 1
        end #end while i

        #apply dynamic regularization if a sign
        #vector has been specified.
        if record_pattern
            rowptr[k + 1] = rowwrite
        end
        if(Dsigns !== nothing && Dsigns[k]*D[k] < regularize_eps)
            D[k] = regularize_delta * Dsigns[k]
            regularize_count[1] += 1
        end

        #Maintain a count of the positive entries
        #in D.  If we hit a zero, we can't factor
        #this matrix, so abort
        if(D[k] == zeroT) return -1 end
        if(D[k]  > zeroT) positiveValuesInD += 1 end

        #compute the inverse of the diagonal
        Dinv[k]= 1/D[k]

        k += 1
    end #end while k

    return positiveValuesInD

end

function QDLDL_numeric_factor!(
        n,
        Ap,
        Ai,
        Ax,
        Lp,
        Li,
        Lx,
        D,
        Dinv,
        rowptr,
        rowcols,
        rowvals,
        fwork,
        Dsigns,
        regularize_eps,
        regularize_delta,
        regularize_count,
)

    positiveValuesInD = 0
    regularize_count[1] = 0
    yVals = fwork
    zeroT = zero(eltype(D))
    fill!(yVals, zeroT)

    D1 = Ax[1]
    if(Dsigns !== nothing && Dsigns[1] * D1 < regularize_eps)
        D1 = regularize_delta * Dsigns[1]
        regularize_count[1] += 1
    end
    if(D1 == zeroT)
        return -1
    end
    D[1] = D1
    if(D1 > zeroT)
        positiveValuesInD += 1
    end
    Dinv[1] = inv(D1)

    k = 2
    @inbounds while k <= n
        Dk = zeroT
        i = Ap[k]
        istop = Ap[k + 1] - 1
        while i <= istop
            bidx = Ai[i]
            axi = Ax[i]
            if bidx == k
                Dk = axi
            else
                yVals[bidx] = axi
            end
            i += 1
        end

        rowidx = rowptr[k]
        rowstop = rowptr[k + 1] - 1
        while rowidx <= rowstop
            cidx = rowcols[rowidx]
            tmpIdx = rowvals[rowidx]
            yVals_cidx = yVals[cidx]
            j = Lp[cidx]
            jstop = tmpIdx - 1
            while j <= jstop
                rowj = Li[j]
                yVals[rowj] = muladd(-Lx[j], yVals_cidx, yVals[rowj])
                j += 1
            end

            lx = yVals_cidx * Dinv[cidx]
            Lx[tmpIdx] = lx
            Dk = muladd(-yVals_cidx, lx, Dk)
            yVals[cidx] = zeroT
            rowidx += 1
        end

        if(Dsigns !== nothing && Dsigns[k] * Dk < regularize_eps)
            Dk = regularize_delta * Dsigns[k]
            regularize_count[1] += 1
        end
        if(Dk == zeroT)
            return -1
        end
        D[k] = Dk
        if(Dk > zeroT)
            positiveValuesInD += 1
        end
        Dinv[k] = inv(Dk)
        k += 1
    end

    return positiveValuesInD
end

# Solves (L+I)x = b, with x replacing b
function QDLDL_Lsolve!(n,Lp,Li,Lx,x)

    i = 1
    @inbounds while i <= n
        xi = x[i]
        j = Lp[i]
        jstop = Lp[i + 1] - 1
        while j <= jstop
            rowj = Li[j]
            x[rowj] = muladd(-Lx[j], xi, x[rowj])
            j += 1
        end
        i += 1
    end
    return nothing
end


# Solves (L+I)'x = b, with x replacing b
function QDLDL_Ltsolve!(n,Lp,Li,Lx,x)

    i = n
    @inbounds while i >= 1
        j = Lp[i]
        jstop = Lp[i + 1] - 1
        xi = x[i]
        while j <= jstop
            xi = muladd(-Lx[j], x[Li[j]], xi)
            j += 1
        end
        x[i] = xi
        i -= 1
    end
    return nothing
end

# Solves Ax = b where A has given LDL factors,
# with x replacing b
function QDLDL_solve!(n,Lp,Li,Lx,Dinv,b)

    QDLDL_Lsolve!(n,Lp,Li,Lx,b)
    @inbounds @simd for i in eachindex(b, Dinv)
        b[i] *= Dinv[i]
    end
    QDLDL_Ltsolve!(n,Lp,Li,Lx,b)

    return nothing
end



# internal permutation and inverse permutation
# functions that require no memory allocations
function permute!(x,b,p)
    @inbounds @simd for j in eachindex(x, p)
        x[j] = b[p[j]]
    end
    return x
end

function ipermute!(x,b,p)
    @inbounds for j in eachindex(x, p)
        x[p[j]] = b[j]
    end
    return x
end

function inverse_permute!(x, b, iperm)
    @inbounds @simd for j in eachindex(x, iperm)
        x[j] = b[iperm[j]]
    end
    return x
end


"Given a sparse symmetric matrix `A` (with only upper triangular entries), return permuted sparse symmetric matrix `P` (only upper triangular) given the inverse permutation vector `iperm`."
function permute_symmetric(
    A::SparseMatrixCSC{Tf, Ti},
    iperm::AbstractVector{Ti},
    Pr::AbstractVector{Ti} = zeros(Ti, nnz(A)),
    Pc::AbstractVector{Ti} = zeros(Ti, size(A, 1) + 1),
    Pv::AbstractVector{Tf} = zeros(Tf, nnz(A))
) where {Tf <: AbstractFloat, Ti <: Integer}

    # perform a number of argument checks
    m, n = size(A)
    m != n && throw(DimensionMismatch("Matrix A must be sparse and square"))

    isperm(iperm) || throw(ArgumentError("pinv must be a permutation"))

    if n != length(iperm)
        throw(DimensionMismatch("Dimensions of sparse matrix A must equal the length of iperm, $((m,n)) != $(iperm)"))
    end

    #we will record a mapping of entries from A to PAPt
    AtoPAPt = zeros(Ti,length(Pv))

    P = _permute_symmetric(A, AtoPAPt, iperm, Pr, Pc, Pv)
    return P, AtoPAPt
end

# the main function without extra argument checks
# following the book: Timothy Davis - Direct Methods for Sparse Linear Systems
function _permute_symmetric(
    A::SparseMatrixCSC{Tf, Ti},
    AtoPAPt::AbstractVector{Ti},
    iperm::AbstractVector{Ti},
    Pr::AbstractVector{Ti},
    Pc::AbstractVector{Ti},
    Pv::AbstractVector{Tf}
) where {Tf <: AbstractFloat, Ti <: Integer}

    # 1. count number of entries that each column of P will have
    n = size(A, 2)
    num_entries = zeros(Ti, n)
    Ar = A.rowval
    Ac = A.colptr
    Av = A.nzval

    # count the number of upper-triangle entries in columns of P, keeping in mind the row permutation
    for colA = 1:n
        colP = iperm[colA]
        # loop over entries of A in column A...
        for row_idx = Ac[colA]:Ac[colA+1]-1
            rowA = Ar[row_idx]
            rowP = iperm[rowA]
            # ...and check if entry is upper triangular
            if rowA <= colA
                # determine to which column the entry belongs after permutation
                col_idx = max(rowP, colP)
                num_entries[col_idx] += one(Ti)
            end
        end
    end
    # 2. calculate permuted Pc = P.colptr from number of entries
    Pc[1] = one(Ti)
    @inbounds for k = 1:n
        Pc[k + 1] = Pc[k] + num_entries[k]

        # reuse this vector memory to keep track of free entries in rowval
        num_entries[k] = Pc[k]
    end
    # use alias
    row_starts = num_entries

    # 3. permute the row entries and position of corresponding nzval
    for colA = 1:n
        colP = iperm[colA]
        # loop over rows of A and determine where each row entry of A should be stored
        for rowA_idx = Ac[colA]:Ac[colA+1]-1
            rowA = Ar[rowA_idx]
            # check if upper triangular
            if rowA <= colA
                rowP = iperm[rowA]
                # determine column to store the entry
                col_idx = max(colP, rowP)

                # find next free location in rowval (this results in unordered columns in the rowval)
                rowP_idx = row_starts[col_idx]

                # store rowval and nzval
                Pr[rowP_idx] = min(colP, rowP)
                Pv[rowP_idx] = Av[rowA_idx]

                #record this into the mapping vector
                AtoPAPt[rowA_idx] = rowP_idx

                # increment next free location
                row_starts[col_idx] += 1
            end
        end
    end
    nz_new = Pc[end] - 1
    P = SparseMatrixCSC{Tf, Ti}(n, n, Pc, Pr[1:nz_new], Pv[1:nz_new])

    return P
end

function _get_amd_ordering(A,amd_dense_scale)

    # PJG: For interested readers - setting amd_dense_scale to 1.5 seems to work better
    # for KKT systems in QP problems, but this ad hoc method can surely be improved

    # computes a permutation for A using AMD default parameters explicit cast of the scaling 
    # to Float64 here allows the scale parameter to be passed as some other float type for 
    # consistency with the main API.

    meta = Amd()
    meta.control[AMD.AMD_DENSE] *= Float64(amd_dense_scale)   
    p = amd(A,meta)
    return p



end

end #end module

