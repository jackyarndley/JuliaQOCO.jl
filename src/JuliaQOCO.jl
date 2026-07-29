module JuliaQOCO

using LinearAlgebra
using PrecompileTools: @compile_workload
using Printf
using SparseArrays

import MathOptInterface as MOI

include("internal_qdldl.jl")
const QDLDL = InternalQDLDL

const MOIU = MOI.Utilities

export Optimizer

include("statuscodes.jl")
include("settings.jl")
include("types.jl")
include("utils.jl")
include("common_linalg.jl")
include("equilibration.jl")
include("cones.jl")
include("kkt.jl")
include("solver.jl")
include("updates.jl")
include("moi_wrapper.jl")

default_settings(::Type{T} = Float64) where {T<:AbstractFloat} = Settings{T}()

moi_version() = string(pkgversion(MOI))

@compile_workload begin
    settings = Settings{Float64}(; verbose = false)
    P = spzeros(Float64, 1, 1)
    G = sparse([-1.0;;])
    solver = Solver(P, [1.0], nothing, nothing, G, [-1.0], 1, Int[]; settings = settings)
    solve!(solver)
    update_vector_data!(solver; c = [0.5])
    solve!(solver)
end

end
