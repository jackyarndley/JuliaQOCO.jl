struct ScalingStats{T<:AbstractFloat}
    obj_range_min::T
    obj_range_max::T
    constraint_range_min::T
    constraint_range_max::T
    rhs_range_min::T
    rhs_range_max::T
end

function ScalingStats(::Type{T}) where {T<:AbstractFloat}
    z = zero(T)
    return ScalingStats{T}(z, z, z, z, z, z)
end

mutable struct ProblemData{T<:AbstractFloat,Ti<:Integer}
    P::SparseMatrixCSC{T,Ti}
    c::Vector{T}
    A::SparseMatrixCSC{T,Ti}
    At::SparseMatrixCSC{T,Ti}
    b::Vector{T}
    G::SparseMatrixCSC{T,Ti}
    Gt::SparseMatrixCSC{T,Ti}
    h::Vector{T}
    l::Ti
    q::Vector{Ti}
    n::Ti
    m::Ti
    p::Ti
    stats::ScalingStats{T}
end

mutable struct Scaling{T<:AbstractFloat}
    delta::Vector{T}
    Druiz::Vector{T}
    Eruiz::Vector{T}
    Fruiz::Vector{T}
    Dinvruiz::Vector{T}
    Einvruiz::Vector{T}
    Finvruiz::Vector{T}
    k::T
    kinv::T
end

mutable struct Workspace{T<:AbstractFloat,Ti<:Integer}
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}
    mu::T
    a::T
    sigma::T
    WtW::Vector{T}
    Wfull::Vector{T}
    Winvfull::Vector{T}
    lambda::Vector{T}
    sbar::Vector{T}
    zbar::Vector{T}
    xbuff::Vector{T}
    ybuff::Vector{T}
    ubuff1::Vector{T}
    ubuff2::Vector{T}
    ubuff3::Vector{T}
    Ds::Vector{T}
    rhs::Vector{T}
    xyz::Vector{T}
    xyzbuff1::Vector{T}
    xyzbuff2::Vector{T}
    kktres::Vector{T}
    soc_offsets::Vector{Ti}
    Wfull_offsets::Vector{Ti}
    Wtri_offsets::Vector{Ti}
end

mutable struct Solution{T<:AbstractFloat}
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}
    iters::Int
    setup_time_sec::Float64
    solve_time_sec::Float64
    obj::T
    pres::T
    dres::T
    gap::T
    status::SolveStatus
end

function Solution(::Type{T}, n::Integer, m::Integer, p::Integer) where {T<:AbstractFloat}
    z = zero(T)
    return Solution{T}(zeros(T, n), zeros(T, m), zeros(T, p), zeros(T, m), 0, 0.0, 0.0, z, z, z, z, QOCO_UNSOLVED)
end

mutable struct LinearSystem{T<:AbstractFloat,Ti<:Integer,F}
    factor::F
    nt2kkt::Vector{Ti}
    ntdiag_positions::Vector{Ti}
    nt_values::Vector{T}
end

mutable struct Solver{T<:AbstractFloat,Ti<:Integer,F}
    settings::Settings{T}
    data::ProblemData{T,Ti}
    scaling::Scaling{T}
    work::Workspace{T,Ti}
    linsys::LinearSystem{T,Ti,F}
    solution::Solution{T}
end
