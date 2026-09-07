using Test

@testset "JuliaQOCO" begin
    selected = isempty(ARGS) ? Set(["native", "numerics", "moi"]) : Set(ARGS)
    if "native" in selected
        include("native_tests.jl")
    end
    if "numerics" in selected
        include("numerics_tests.jl")
    end
    if "moi" in selected
        include("moi_wrapper_tests.jl")
        include("moi_semantics_tests.jl")
        include("moi_test_subset.jl")
    end
end
