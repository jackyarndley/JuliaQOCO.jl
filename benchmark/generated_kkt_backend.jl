using BenchmarkTools
using SparseArrays

using JuliaQOCO

include(joinpath(@__DIR__, "scp_like_benchmark.jl"))
include(joinpath(@__DIR__, "..", "examples", "scp_jump_reuse.jl"))

function build_backend_solver(
    backend::Symbol;
    horizon::Int,
    nx::Int,
    nu::Int,
    max_iters::Int,
)
    P, c, A, b, G, h, q, case = make_benchmark_case(;
        horizon = horizon,
        nx = nx,
        nu = nu,
    )
    settings = JuliaQOCO.Settings{Float64}(;
        verbose = false,
        profile = false,
        max_iters = max_iters,
        kkt_backend = backend,
        scaling_mode = :once,
        warm_start_mode = :primal_dual,
    )
    wall_setup_sec = @elapsed solver = JuliaQOCO.Solver(
        P,
        c,
        A,
        b,
        G,
        h,
        0,
        q;
        settings = settings,
    )
    return solver, case, wall_setup_sec
end

function benchmark_generated_backend_case(;
    horizon::Int,
    nx::Int = 6,
    nu::Int = 3,
    max_iters::Int = 20,
    samples::Int = 100,
)
    standard, standard_case, standard_setup_wall = build_backend_solver(
        :qdldl;
        horizon,
        nx,
        nu,
        max_iters,
    )
    generated, generated_case, generated_setup_wall = build_backend_solver(
        :generated;
        horizon,
        nx,
        nu,
        max_iters,
    )
    _, _, generated_reused_setup_wall = build_backend_solver(
        :generated;
        horizon,
        nx,
        nu,
        max_iters,
    )

    # Warm the complete solve/update paths. Pattern generation and compilation
    # have already been charged to generated_setup_wall.
    prepare_cached_solver!(standard, standard_case)
    prepare_cached_solver!(generated, generated_case)

    standard_factor = @benchmark JuliaQOCO.QDLDL.refactor!(
        $standard.linsys.factor,
    ) evals = 1 samples = samples
    generated_factor = @benchmark JuliaQOCO.QDLDL.refactor!(
        $generated.linsys.factor,
    ) evals = 1 samples = samples
    standard_rhs = copy(standard.work.rhs)
    generated_rhs = copy(generated.work.rhs)
    standard_direction = similar(standard_rhs)
    generated_direction = similar(generated_rhs)
    standard_triangular = @benchmark JuliaQOCO.QDLDL.solve!(
        $standard.linsys.factor,
        $standard_direction,
    ) setup = (copyto!($standard_direction, $standard_rhs)) evals = 1 samples = samples
    generated_triangular = @benchmark JuliaQOCO.QDLDL.solve!(
        $generated.linsys.factor,
        $generated_direction,
    ) setup = (copyto!($generated_direction, $generated_rhs)) evals = 1 samples = samples
    standard_product = @benchmark JuliaQOCO.kkt_multiply!(
        $standard_direction,
        $standard_rhs,
        $standard.data,
        $standard.work,
        $standard.linsys.factor.generated_pattern,
    ) evals = 1 samples = samples
    generated_product = @benchmark JuliaQOCO.kkt_multiply!(
        $generated_direction,
        $generated_rhs,
        $generated.data,
        $generated.work,
        $generated.linsys.factor.generated_pattern,
    ) evals = 1 samples = samples

    standard_solve = @benchmark begin
        advance_case!($standard_case)
        JuliaQOCO.update_vector_data!(
            $standard;
            c = $standard_case.ctrial,
            b = $standard_case.btrial,
            h = $standard_case.htrial,
        )
        JuliaQOCO.update_matrix_data!(
            $standard;
            Ax = $standard_case.Axtrial,
            Gx = $standard_case.Gxtrial,
        )
        JuliaQOCO.solve!($standard)
    end evals = 1 samples = samples
    generated_solve = @benchmark begin
        advance_case!($generated_case)
        JuliaQOCO.update_vector_data!(
            $generated;
            c = $generated_case.ctrial,
            b = $generated_case.btrial,
            h = $generated_case.htrial,
        )
        JuliaQOCO.update_matrix_data!(
            $generated;
            Ax = $generated_case.Axtrial,
            Gx = $generated_case.Gxtrial,
        )
        JuliaQOCO.solve!($generated)
    end evals = 1 samples = samples

    sf = median(standard_factor)
    gf = median(generated_factor)
    st = median(standard_triangular)
    gt = median(generated_triangular)
    sp = median(standard_product)
    gp = median(generated_product)
    ss = median(standard_solve)
    gs = median(generated_solve)
    factor = standard.linsys.factor
    println(
        join(
            (
                "horizon=$(horizon)",
                "dimension=$(factor.workspace.Ln)",
                "kkt_nnz=$(nnz(factor.workspace.triuA))",
                "factor_nnz=$(nnz(factor.L))",
                "standard_setup_ms=$(round(1e3 * standard_setup_wall; digits = 3))",
                "generated_setup_ms=$(round(1e3 * generated_setup_wall; digits = 3))",
                "generated_same_pattern_setup_ms=$(round(1e3 * generated_reused_setup_wall; digits = 3))",
                "standard_refactor_us=$(round(sf.time / 1e3; digits = 3))",
                "generated_refactor_us=$(round(gf.time / 1e3; digits = 3))",
                "refactor_speedup=$(round(sf.time / gf.time; digits = 3))",
                "standard_triangular_us=$(round(st.time / 1e3; digits = 3))",
                "generated_triangular_us=$(round(gt.time / 1e3; digits = 3))",
                "triangular_speedup=$(round(st.time / gt.time; digits = 3))",
                "standard_kkt_product_us=$(round(sp.time / 1e3; digits = 3))",
                "generated_kkt_product_us=$(round(gp.time / 1e3; digits = 3))",
                "kkt_product_speedup=$(round(sp.time / gp.time; digits = 3))",
                "standard_resolve_ms=$(round(ss.time / 1e6; digits = 3))",
                "generated_resolve_ms=$(round(gs.time / 1e6; digits = 3))",
                "resolve_speedup=$(round(ss.time / gs.time; digits = 3))",
                "standard_allocs=$(ss.allocs)",
                "generated_allocs=$(gs.allocs)",
            ),
            " ",
        ),
    )
    return nothing
end

function run_generated_backend_benchmarks(;
    horizons = (4, 8, 24),
    nx::Int = 6,
    nu::Int = 3,
    max_iters::Int = 20,
    samples::Int = 100,
)
    println("JuliaQOCO generated fixed-pattern KKT benchmark")
    println("nx=$(nx) nu=$(nu) max_iters=$(max_iters) samples=$(samples)")
    for horizon in horizons
        benchmark_generated_backend_case(;
            horizon,
            nx,
            nu,
            max_iters,
            samples,
        )
    end
    return nothing
end

function benchmark_generated_jump_backend(;
    horizon::Int = 8,
    nx::Int = 4,
    nu::Int = 2,
    samples::Int = 50,
)
    # Compile JuMP/MOI assembly and solve machinery on a different pattern so
    # the reported initial solves do not include generic Julia compilation.
    warmup = build_jump_scp_case(;
        horizon = 2,
        nx = 2,
        nu = 1,
        kkt_backend = :generated,
    )
    optimize!(warmup.model)

    standard = build_jump_scp_case(;
        horizon,
        nx,
        nu,
        kkt_backend = :qdldl,
    )
    generated = build_jump_scp_case(;
        horizon,
        nx,
        nu,
        kkt_backend = :generated,
    )
    standard_initial = @elapsed optimize!(standard.model)
    generated_initial = @elapsed optimize!(generated.model)
    update_jump_scp!(standard)
    update_jump_scp!(generated)
    optimize!(standard.model)
    optimize!(generated.model)

    standard_trial = @benchmark begin
        update_jump_scp!($standard)
        optimize!($standard.model)
    end evals = 1 samples = samples
    generated_trial = @benchmark begin
        update_jump_scp!($generated)
        optimize!($generated.model)
    end evals = 1 samples = samples
    standard_estimate = median(standard_trial)
    generated_estimate = median(generated_trial)
    standard_solver = JuMP.MOI.get(
        JuMP.backend(standard.model),
        JuMP.MOI.RawSolver(),
    )
    generated_solver = JuMP.MOI.get(
        JuMP.backend(generated.model),
        JuMP.MOI.RawSolver(),
    )
    objective_error = abs(
        objective_value(standard.model) - objective_value(generated.model),
    )
    primal_error = maximum(
        abs,
        value.(standard.x) .- value.(generated.x),
    )
    println(
        join(
            (
                "jump_horizon=$(horizon)",
                "nx=$(nx)",
                "nu=$(nu)",
                "standard_initial_ms=$(round(1e3 * standard_initial; digits = 3))",
                "generated_initial_ms=$(round(1e3 * generated_initial; digits = 3))",
                "standard_repeated_ms=$(round(standard_estimate.time / 1e6; digits = 3))",
                "generated_repeated_ms=$(round(generated_estimate.time / 1e6; digits = 3))",
                "repeated_speedup=$(round(standard_estimate.time / generated_estimate.time; digits = 3))",
                "standard_alloc_kib=$(round(standard_estimate.memory / 1024; digits = 3))",
                "generated_alloc_kib=$(round(generated_estimate.memory / 1024; digits = 3))",
                "standard_native_setup_ms=$(round(1e3 * standard_solver.solution.setup_time_sec; digits = 3))",
                "generated_native_setup_ms=$(round(1e3 * generated_solver.solution.setup_time_sec; digits = 3))",
                "objective_error=$(objective_error)",
                "max_primal_error=$(primal_error)",
                "standard_iters=$(standard_solver.solution.iters)",
                "generated_iters=$(generated_solver.solution.iters)",
            ),
            " ",
        ),
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_generated_backend_benchmarks()
    benchmark_generated_jump_backend()
end
