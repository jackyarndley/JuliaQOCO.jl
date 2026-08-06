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
    return settings
end
