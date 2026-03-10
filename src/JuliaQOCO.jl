module JuliaQOCO

using LinearAlgebra
using Printf
using SparseArrays

import MathOptInterface as MOI
import QDLDL

const MOIU = MOI.Utilities

export Settings,
       SolveStatus,
       Solution,
       Solver,
       Optimizer,
       copy_settings,
       default_settings,
       solve!,
       solve

include("statuscodes.jl")
include("settings.jl")
include("types.jl")
include("utils.jl")
include("common_linalg.jl")
include("equilibration.jl")
include("cones.jl")
include("kkt.jl")
include("solver.jl")
include("moi_wrapper.jl")

default_settings(::Type{T} = Float64) where {T<:AbstractFloat} = Settings{T}()

moi_version() = string(pkgversion(MOI))

end
