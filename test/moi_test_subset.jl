import MathOptInterface as MOI
using JuliaQOCO

# Attributes this solver genuinely does not produce. They are excluded by name
# rather than left to the fallback model, which would claim to support them.
const MOI_EXCLUDED_ATTRIBUTES = Any[
    # An interior-point method has no basis.
    MOI.VariableBasisStatus,
    MOI.ConstraintBasisStatus,
    # No bound or dual objective is certified; see the note on certificates
    # below.
    MOI.ObjectiveBound,
    MOI.DualObjectiveValue,
    # Names and deletion are exercised through the fallback model rather than
    # the numerical layer.
    MOI.VariableName,
    MOI.ConstraintName,
    MOI.delete,
]

const MOI_TEST_CONFIG = MOI.Test.Config(
    Float64;
    atol = 1e-4,
    rtol = 1e-4,
    exclude = MOI_EXCLUDED_ATTRIBUTES,
)

# Tests excluded for reasons that are properties of the algorithm, not gaps in
# the wrapper. Every entry is a case that requires an infeasibility or
# unboundedness certificate: this solver reports an honest numerical failure
# when a subproblem has no solution, and deliberately does not manufacture a
# certificate it has not computed.
const MOI_EXCLUDED_TESTS = [
    r"INFEASIBLE",
    r"INFEASIBILITY_CERTIFICATE",
    # These three check that an unbounded second-order cone problem is
    # reported as DUAL_INFEASIBLE.
    "test_conic_SecondOrderCone_negative_post_bound",
    "test_conic_SecondOrderCone_no_initial_bound",
    # The editable model is a UniversalFallback, which by design accepts any
    # attribute, so no UnsupportedAttribute is ever raised.
    "test_model_copy_to_UnsupportedAttribute",
]

# The families of conformance tests that apply to a continuous conic solver.
# Integer, nonlinear and constraint-programming families are not listed
# because the solver does not support those problem classes at all.
const MOI_TEST_FAMILIES = [
    r"^test_attribute_",
    r"^test_basic_",
    r"^test_conic_",
    r"^test_linear_",
    r"^test_model_",
    r"^test_modification_",
    r"^test_objective_",
    r"^test_quadratic_",
    r"^test_solve_",
    r"^test_variable_",
]

# Full conformance run through the standard bridge and cache layers.
MOI.Test.runtests(
    MOI.instantiate(
        JuliaQOCO.Optimizer;
        with_bridge_type = Float64,
        with_cache_type = Float64,
    ),
    MOI_TEST_CONFIG;
    include = MOI_TEST_FAMILIES,
    exclude = MOI_EXCLUDED_TESTS,
    warn_unsupported = false,
)

# A smaller run straight against the optimizer, with no bridges and no caching
# layer, which is how `JuMP.direct_model` uses it.
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
        "test_solve_result_index",
        "test_modification_func_scalaraffine_lessthan",
        "test_modification_func_vectoraffine_nonneg",
        "test_modification_set_scalaraffine_lessthan",
        "test_modification_coef_scalar_objective",
        "test_modification_objective_scalarquadraticcoefficientchange",
        "test_objective_set_via_modify",
        "test_quadratic_duplicate_terms",
    ],
    exclude = MOI_EXCLUDED_TESTS,
    warn_unsupported = false,
)
