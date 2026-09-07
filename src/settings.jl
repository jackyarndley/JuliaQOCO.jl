Base.@kwdef mutable struct Settings{T<:AbstractFloat}
    max_iters::Int = 200
    bisect_iters::Int = 5
    ruiz_iters::Int = 3
    iter_ref_iters::Int = 1
    iter_ref_tol::T = sqrt(eps(T))
    kkt_static_reg::T = T(1e-8)
    kkt_dynamic_reg::T = T(1e-8)
    abstol::T = T(1e-7)
    reltol::T = T(1e-7)
    abstol_inacc::T = T(1e-5)
    reltol_inacc::T = T(1e-5)
    verbose::Bool = true
    profile::Bool = false
    reuse_solver::Bool = true
    scaling_mode::Symbol = :once
    warm_start_mode::Symbol = :primal_dual
    # Convexity validation of the quadratic objective.
    #   :auto - verify on construction and on every update that changes P,
    #           using the cheapest conclusive test for the pattern at hand.
    #   :none - the caller guarantees that P is positive semidefinite. No test
    #           is run and no claim of verified convexity is made.
    convexity_check::Symbol = :auto
    # Largest dimension for which :auto uses a dense eigenvalue decomposition.
    # Above it, a shifted sparse Cholesky provides the same guarantee at a far
    # lower cost. Only relevant for a Hessian that is not diagonal.
    convexity_dense_limit::Int = 512
    # Wall-clock budget for one solve, in seconds. Checked at iteration
    # boundaries only: a single factorization call is not preemptible, so this
    # is a budget and not a hard real-time deadline.
    time_limit_sec::Float64 = Inf
    # Second-order cones of at least this dimension get the sparse
    # auxiliary-variable expansion of their Nesterov-Todd block instead of a
    # dense upper-triangular one. The expansion is exact, not an approximation:
    # see `docs/internals.md`. The default sits above the measured crossover,
    # which is near dimension ten; below it the compact dense block is faster,
    # above it the expansion wins by a margin that grows with the dimension.
    # Set to `typemax(Int)` to disable the expansion entirely.
    soc_expansion_threshold::Int = 16
    output::IO = stdout
end

function copy_settings(settings::Settings{T}) where {T<:AbstractFloat}
    return Settings{T}(;
        max_iters = settings.max_iters,
        bisect_iters = settings.bisect_iters,
        ruiz_iters = settings.ruiz_iters,
        iter_ref_iters = settings.iter_ref_iters,
        iter_ref_tol = settings.iter_ref_tol,
        kkt_static_reg = settings.kkt_static_reg,
        kkt_dynamic_reg = settings.kkt_dynamic_reg,
        abstol = settings.abstol,
        reltol = settings.reltol,
        abstol_inacc = settings.abstol_inacc,
        reltol_inacc = settings.reltol_inacc,
        verbose = settings.verbose,
        profile = settings.profile,
        reuse_solver = settings.reuse_solver,
        scaling_mode = settings.scaling_mode,
        warm_start_mode = settings.warm_start_mode,
        convexity_check = settings.convexity_check,
        convexity_dense_limit = settings.convexity_dense_limit,
        time_limit_sec = settings.time_limit_sec,
        soc_expansion_threshold = settings.soc_expansion_threshold,
        output = settings.output,
    )
end

function validate_settings(settings::Settings)
    settings.max_iters > 0 || throw(ArgumentError("max_iters must be positive"))
    settings.ruiz_iters >= 0 || throw(ArgumentError("ruiz_iters must be nonnegative"))
    settings.bisect_iters > 0 || throw(ArgumentError("bisect_iters must be positive"))
    settings.iter_ref_iters >= 0 || throw(ArgumentError("iter_ref_iters must be nonnegative"))
    settings.iter_ref_tol >= 0 || throw(ArgumentError("iter_ref_tol must be nonnegative"))
    settings.abstol > 0 || throw(ArgumentError("abstol must be positive"))
    settings.reltol >= 0 || throw(ArgumentError("reltol must be nonnegative"))
    settings.abstol_inacc > 0 || throw(ArgumentError("abstol_inacc must be positive"))
    settings.reltol_inacc >= 0 || throw(ArgumentError("reltol_inacc must be nonnegative"))
    settings.kkt_static_reg > 0 || throw(ArgumentError("kkt_static_reg must be positive"))
    settings.kkt_dynamic_reg > 0 || throw(ArgumentError("kkt_dynamic_reg must be positive"))
    settings.scaling_mode in (:none, :once, :recompute) ||
        throw(ArgumentError("scaling_mode must be :none, :once, or :recompute"))
    settings.warm_start_mode in (:none, :primal, :primal_dual, :adaptive) ||
        throw(ArgumentError("warm_start_mode must be :none, :primal, :primal_dual, or :adaptive"))
    settings.convexity_check in (:auto, :none) ||
        throw(ArgumentError("convexity_check must be :auto or :none"))
    settings.convexity_dense_limit >= 0 ||
        throw(ArgumentError("convexity_dense_limit must be nonnegative"))
    (settings.time_limit_sec >= 0 && !isnan(settings.time_limit_sec)) ||
        throw(ArgumentError("time_limit_sec must be nonnegative"))
    settings.soc_expansion_threshold >= 2 ||
        throw(ArgumentError("soc_expansion_threshold must be at least two"))
    isfinite(settings.iter_ref_tol) || throw(ArgumentError("iter_ref_tol must be finite"))
    (isfinite(settings.abstol) && isfinite(settings.reltol)) ||
        throw(ArgumentError("tolerances must be finite"))
    (isfinite(settings.abstol_inacc) && isfinite(settings.reltol_inacc)) ||
        throw(ArgumentError("inaccurate tolerances must be finite"))
    (isfinite(settings.kkt_static_reg) && isfinite(settings.kkt_dynamic_reg)) ||
        throw(ArgumentError("regularization parameters must be finite"))
    return settings
end
