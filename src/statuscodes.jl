@enum SolveStatus::Int begin
    QOCO_UNSOLVED = 0
    QOCO_SOLVED = 1
    QOCO_SOLVED_INACCURATE = 2
    QOCO_NUMERICAL_ERROR = 3
    QOCO_MAX_ITER = 4
end

const STATUS_MESSAGES = Dict(
    QOCO_UNSOLVED => "unsolved",
    QOCO_SOLVED => "solved",
    QOCO_SOLVED_INACCURATE => "solved_inaccurately",
    QOCO_NUMERICAL_ERROR => "numerical_error",
    QOCO_MAX_ITER => "max_iterations",
)

status_string(status::SolveStatus, detail::AbstractString = "") =
    isempty(detail) ? STATUS_MESSAGES[status] : string(STATUS_MESSAGES[status], " (", detail, ")")
