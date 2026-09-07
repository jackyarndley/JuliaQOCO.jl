# An independent numerical oracle for JuliaQOCO results.
#
# Everything here is computed from the ORIGINAL, unscaled problem data with
# ordinary linear algebra. Nothing in this file calls the solver residual,
# objective, scaling or stopping functions it exists to validate: if the two
# ever agree, that agreement means something.
#
# The problem form is
#
#     minimize    (1/2) x' P x + c' x
#     subject to  A x = b
#                 h - G x in K
#
# with K the product of the nonnegative orthant of dimension l followed by the
# second-order cones of dimensions q[1], q[2], ... The dual variable of the
# equality block is y and of the cone block is z.

module Oracle

using LinearAlgebra
using SparseArrays
using Test

export OracleProblem, OracleReport, check_result, oracle_report, cone_distance

struct OracleProblem{T<:AbstractFloat}
    P::SparseMatrixCSC{T,Int}   # upper-triangular convention, as passed to the solver
    c::Vector{T}
    A::Union{Nothing,SparseMatrixCSC{T,Int}}
    b::Union{Nothing,Vector{T}}
    G::Union{Nothing,SparseMatrixCSC{T,Int}}
    h::Union{Nothing,Vector{T}}
    l::Int
    q::Vector{Int}
end

function OracleProblem(P, c::Vector{T}, A, b, G, h, l::Integer, q) where {T}
    n = length(c)
    Pm = P === nothing ? spzeros(T, n, n) : SparseMatrixCSC{T,Int}(P)
    return OracleProblem{T}(
        Pm,
        copy(c),
        A === nothing ? nothing : SparseMatrixCSC{T,Int}(A),
        b === nothing ? nothing : Vector{T}(b),
        G === nothing ? nothing : SparseMatrixCSC{T,Int}(G),
        h === nothing ? nothing : Vector{T}(h),
        Int(l),
        Int[qi for qi in q],
    )
end

struct OracleReport{T<:AbstractFloat}
    objective::T
    eq_residual::T
    cone_residual::T
    dual_residual::T
    primal_cone_distance::T
    dual_cone_distance::T
    complementarity::T
    primal_reference::T
    dual_reference::T
    objective_reference::T
    finite::Bool
end

# Distance from u to the product cone: zero inside, positive outside. Written
# directly from the definition rather than reusing any solver kernel.
function cone_distance(u::AbstractVector{T}, l::Integer, q::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    isempty(u) && return zero(T)
    all(isfinite, u) || return T(Inf)
    worst = zero(T)
    for i in 1:l
        worst = max(worst, -u[i])
    end
    first = Int(l) + 1
    for qi in q
        head = u[first]
        tail = norm(view(u, (first + 1):(first + qi - 1)))
        worst = max(worst, tail - head)
        first += qi
    end
    return max(worst, zero(T))
end

# Symmetric Hessian from the upper-triangular storage convention.
function _symmetric_hessian(P::SparseMatrixCSC{T}) where {T}
    n = size(P, 1)
    n <= 400 && return Matrix(Symmetric(Matrix(P), :U))
    return P + transpose(P) - Diagonal(diag(P))
end

function oracle_report(
    problem::OracleProblem{T},
    x::AbstractVector{T},
    s::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
) where {T<:AbstractFloat}
    n = length(problem.c)
    Psym = _symmetric_hessian(problem.P)
    Px = Psym * x
    Aty = problem.A === nothing ? zeros(T, n) : transpose(problem.A) * y
    Gtz = problem.G === nothing ? zeros(T, n) : transpose(problem.G) * z
    Ax = problem.A === nothing ? T[] : problem.A * x
    Gx = problem.G === nothing ? T[] : problem.G * x

    eq_residual = problem.A === nothing ? zero(T) : norm(Ax - problem.b, Inf)
    cone_residual = problem.G === nothing ? zero(T) : norm(Gx + s - problem.h, Inf)
    dual_residual = norm(Px + problem.c + Aty + Gtz, Inf)

    objective = T(0.5) * dot(x, Px) + dot(problem.c, x)
    complementarity = isempty(s) ? zero(T) : dot(s, z)

    primal_reference = max(
        problem.A === nothing ? zero(T) : norm(Ax, Inf),
        problem.b === nothing ? zero(T) : norm(problem.b, Inf),
        problem.G === nothing ? zero(T) : norm(Gx, Inf),
        problem.h === nothing ? zero(T) : norm(problem.h, Inf),
        isempty(s) ? zero(T) : norm(s, Inf),
    )
    dual_reference = max(
        norm(Px, Inf),
        norm(problem.c, Inf),
        norm(Aty, Inf),
        norm(Gtz, Inf),
    )
    bty = problem.b === nothing ? zero(T) : dot(problem.b, y)
    htz = problem.h === nothing ? zero(T) : dot(problem.h, z)
    objective_reference = max(
        one(T),
        abs(objective),
        abs(-T(0.5) * dot(x, Px) - bty - htz),
    )

    finite =
        all(isfinite, x) && all(isfinite, s) && all(isfinite, y) && all(isfinite, z) &&
        isfinite(objective) && isfinite(complementarity)

    return OracleReport{T}(
        objective,
        eq_residual,
        cone_residual,
        dual_residual,
        cone_distance(s, problem.l, problem.q),
        cone_distance(z, problem.l, problem.q),
        complementarity,
        primal_reference,
        dual_reference,
        objective_reference,
        finite,
    )
end

"""
    check_result(problem, solution; atol, rtol, label)

Assert that a returned iterate is a genuine, mutually consistent optimum of
`problem` in original units, and that the metrics the solver reported describe
that same iterate. Every quantity here is recomputed from scratch.
"""
function check_result(
    problem::OracleProblem{T},
    solution;
    atol::T = T(1e-6),
    rtol::T = T(1e-6),
    label::AbstractString = "",
    check_reported::Bool = true,
) where {T<:AbstractFloat}
    report = oracle_report(problem, solution.x, solution.s, solution.y, solution.z)
    prefix = isempty(label) ? "" : "$label: "

    @test report.finite
    # Cone membership, up to the rounding the solver itself tolerates.
    cone_tol = T(1e-8) * max(one(T), report.primal_reference)
    @test report.primal_cone_distance <= cone_tol
    @test report.dual_cone_distance <= cone_tol

    @test report.eq_residual <= atol + rtol * report.primal_reference
    @test report.cone_residual <= atol + rtol * report.primal_reference
    @test report.dual_residual <= atol + rtol * report.dual_reference
    @test abs(report.complementarity) <= atol + rtol * report.objective_reference

    if check_reported
        # The reported metrics must describe the returned vectors. The primal
        # residual the solver reports is the larger of the two primal blocks.
        reported_primal = max(report.eq_residual, report.cone_residual)
        scale = max(one(T), report.primal_reference)
        @test isapprox(solution.pres, reported_primal; atol = 1e-9 * scale, rtol = 1e-5)
        dual_scale = max(one(T), report.dual_reference)
        @test isapprox(solution.dres, report.dual_residual; atol = 1e-9 * dual_scale, rtol = 1e-5)
        gap_scale = max(one(T), report.objective_reference)
        @test isapprox(solution.gap, report.complementarity; atol = 1e-9 * gap_scale, rtol = 1e-5)
        objective_scale = max(one(T), abs(report.objective))
        @test isapprox(solution.obj, report.objective; atol = 1e-9 * objective_scale, rtol = 1e-7)
    end
    if !isempty(prefix) && !report.finite
        @info string(prefix, "oracle report", report)
    end
    return report
end

end # module Oracle
