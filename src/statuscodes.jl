@enum SolveStatus::Int begin
    QOCO_UNSOLVED = 0
    QOCO_SOLVED = 1
    QOCO_SOLVED_INACCURATE = 2
    QOCO_NUMERICAL_ERROR = 3
    QOCO_MAX_ITER = 4
end

const STATUS_MESSAGES = (
    "unsolved",
    "solved",
    "solved_inaccurately",
    "numerical_error",
    "max_iterations",
)

@inline status_message(status::SolveStatus) = @inbounds STATUS_MESSAGES[Int(status) + 1]

status_string(status::SolveStatus, detail::AbstractString = "") =
    isempty(detail) ? status_message(status) : string(status_message(status), " (", detail, ")")
