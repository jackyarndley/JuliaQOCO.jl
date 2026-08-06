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

mutable struct SolveProfile
    problem_data_time_sec::Float64
    workspace_time_sec::Float64
    linsys_time_sec::Float64
    initialize_time_sec::Float64
    residual_time_sec::Float64
    objective_time_sec::Float64
    mu_time_sec::Float64
    stopping_time_sec::Float64
    nt_scaling_time_sec::Float64
    nt_update_time_sec::Float64
    predictor_time_sec::Float64
    linsys_solve_time_sec::Float64
    linsys_refine_time_sec::Float64
    linsys_solves::Int
    linsys_refinements::Int
    nt_refactors::Int
    dynamic_regularizations::Int
end

SolveProfile() = SolveProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

mutable struct ProblemData{T<:AbstractFloat,Ti<:Integer}
    P::SparseMatrixCSC{T,Ti}
    Pcol::Vector{Ti}
    c::Vector{T}
    A::SparseMatrixCSC{T,Ti}
    Acol::Vector{Ti}
    At::SparseMatrixCSC{T,Ti}
    AtoAt::Vector{Ti}
    AfromAt::Vector{Ti}
    b::Vector{T}
    G::SparseMatrixCSC{T,Ti}
    Gcol::Vector{Ti}
    Gt::SparseMatrixCSC{T,Ti}
    GtoGt::Vector{Ti}
    GfromGt::Vector{Ti}
    h::Vector{T}
    l::Ti
    q::Vector{Ti}
    n::Ti
    m::Ti
    p::Ti
    Padded_idx::Vector{Ti}
    stats::ScalingStats{T}
    stats_dirty::Bool
end

mutable struct Scaling{T<:AbstractFloat}
    delta::Vector{T}
    Druiz::Vector{T}
    Eruiz::Vector{T}
    Fruiz::Vector{T}
    Dinvruiz::Vector{T}
    Einvruiz::Vector{T}
    Finvruiz::Vector{T}
    Anorm::Vector{T}
    Gnorm::Vector{T}
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
    nt_v::Vector{T}
    nt_scale::Vector{T}
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
    Wtri_offsets::Vector{Ti}
    quad_obj::T
    xPx::T
    Pxinf::T
    Atyinf::T
    Gtzinf::T
    Axinf::T
    Gxinf::T
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
    status_detail::String
    profile::SolveProfile
    best_x::Vector{T}
    best_s::Vector{T}
    best_y::Vector{T}
    best_z::Vector{T}
    best_metric::T
    best_valid::Bool
end

function Solution(::Type{T}, n::Integer, m::Integer, p::Integer) where {T<:AbstractFloat}
    z = zero(T)
    return Solution{T}(
        zeros(T, n), zeros(T, m), zeros(T, p), zeros(T, m), 0, 0.0, 0.0,
        z, z, z, z, QOCO_UNSOLVED, "", SolveProfile(),
        zeros(T, n), zeros(T, m), zeros(T, p), zeros(T, m), floatmax(T), false,
    )
end

const QDLDLFactor{T,Ti} = Union{
    Nothing,
    QDLDL.QDLDLFactorisation{
        T,
        Ti,
        Vector{Ti},
        Vector{Ti},
        QDLDL.QDLDLWorkspace{T,Ti,Vector{Ti},Vector{Ti}},
    },
}

mutable struct LinearSystem{T<:AbstractFloat,Ti<:Integer}
    factor::QDLDLFactor{T,Ti}
    nt2kkt::Vector{Ti}
    ntdiag_positions::Vector{Ti}
    nt_values::Vector{T}
    static2kkt::Vector{Ti}
    static_values::Vector{T}
end

mutable struct Warmstart{T<:AbstractFloat}
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}
    active::Bool
    manual::Bool
    scaled::Bool
    repair::Bool
end

function Warmstart(::Type{T}, n::Integer, m::Integer, p::Integer) where {T<:AbstractFloat}
    return Warmstart{T}(
        zeros(T, n),
        zeros(T, m),
        zeros(T, p),
        zeros(T, m),
        false,
        false,
        false,
        false,
    )
end

mutable struct CoreSolver{T<:AbstractFloat,Ti<:Integer}
    settings::Settings{T}
    data::ProblemData{T,Ti}
    scaling::Scaling{T}
    work::Workspace{T,Ti}
    linsys::LinearSystem{T,Ti}
    solution::Solution{T}
    warmstart::Warmstart{T}
end
