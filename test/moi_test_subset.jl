import MathOptInterface as MOI
using JuliaQOCO

const MOI_TEST_CONFIG = MOI.Test.Config(
    Float64;
    atol = 1e-4,
    rtol = 1e-4,
    exclude = Any[
        MOI.ConstraintDual,
        MOI.DualObjectiveValue,
        MOI.VariableName,
        MOI.ConstraintName,
        MOI.delete,
    ],
)

MOI.Test.runtests(
    JuliaQOCO.Optimizer(; verbose = false),
    MOI_TEST_CONFIG;
    include = [
        "test_model_empty",
        "test_variable_add_variable",
        "test_constraint_ScalarAffineFunction_EqualTo",
        "test_constraint_ScalarAffineFunction_LessThan",
        "test_conic_SecondOrderCone_VectorAffineFunction",
        "test_conic_SecondOrderCone_VectorOfVariables",
        "test_objective_ObjectiveFunction_VariableIndex",
        "test_linear_FEASIBILITY_SENSE",
        "test_solve_optimize_twice",
        "test_modification_func_scalaraffine_lessthan",
        "test_modification_func_vectoraffine_nonneg",
        "test_modification_set_scalaraffine_lessthan",
        "test_modification_coef_scalar_objective",
        "test_objective_set_via_modify",
    ],
    warn_unsupported = false,
)
