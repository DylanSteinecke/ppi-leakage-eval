"""Task-aware split-strategy compatibility metadata."""

from dataclasses import dataclass


PROVIDED_SPLIT_STRATEGY = "provided_column"
RANDOM_SPLIT_STRATEGY = "random"
C1_SPLIT_STRATEGY = "c1"
C2_SPLIT_STRATEGY = "c2"
C3_SPLIT_STRATEGY = "c3"

PROTEIN_DISJOINT_SPLIT_STRATEGIES = {
    C1_SPLIT_STRATEGY,
    C2_SPLIT_STRATEGY,
    C3_SPLIT_STRATEGY,
}
SPLIT_STRATEGY_CHOICES = (
    RANDOM_SPLIT_STRATEGY,
    C1_SPLIT_STRATEGY,
    C2_SPLIT_STRATEGY,
    C3_SPLIT_STRATEGY,
)


@dataclass(frozen=True)
class SplitStrategySpec:
    """Scientific identity and compatibility metadata for one strategy."""

    task_id: str
    strategy_name: str
    protocol_id: str
    protocol_version: int
    prediction_unit: str
    assignment_entity: str
    projection_kind: str
    default_grouping_kind: str | None = None
    allowed_grouping_kinds: tuple[str, ...] = ()
    projection_drop_reasons: tuple[str, ...] = ()


_PPI_TASK_ID = "ppi"
_CANONICAL_UNORDERED_PAIR = "canonical_unordered_protein_pair"
_PROTEIN_GROUPING_KINDS = ("protein_identity", "sequence_cluster")

_SPLIT_STRATEGIES = {
    (_PPI_TASK_ID, PROVIDED_SPLIT_STRATEGY): SplitStrategySpec(
        task_id=_PPI_TASK_ID,
        strategy_name=PROVIDED_SPLIT_STRATEGY,
        protocol_id="ppi.provided.v1",
        protocol_version=1,
        prediction_unit=_CANONICAL_UNORDERED_PAIR,
        assignment_entity="source_assignment",
        projection_kind="source_defined",
    ),
    (_PPI_TASK_ID, RANDOM_SPLIT_STRATEGY): SplitStrategySpec(
        task_id=_PPI_TASK_ID,
        strategy_name=RANDOM_SPLIT_STRATEGY,
        protocol_id="ppi.random_pair.v1",
        protocol_version=1,
        prediction_unit=_CANONICAL_UNORDERED_PAIR,
        assignment_entity=_CANONICAL_UNORDERED_PAIR,
        projection_kind="random_pair_partition",
    ),
    (_PPI_TASK_ID, C1_SPLIT_STRATEGY): SplitStrategySpec(
        task_id=_PPI_TASK_ID,
        strategy_name=C1_SPLIT_STRATEGY,
        protocol_id="ppi.c1.v1",
        protocol_version=1,
        prediction_unit=_CANONICAL_UNORDERED_PAIR,
        assignment_entity=_CANONICAL_UNORDERED_PAIR,
        projection_kind="ppi_c1_edge_holdout",
    ),
    (_PPI_TASK_ID, C2_SPLIT_STRATEGY): SplitStrategySpec(
        task_id=_PPI_TASK_ID,
        strategy_name=C2_SPLIT_STRATEGY,
        protocol_id="ppi.c2.v1",
        protocol_version=1,
        prediction_unit=_CANONICAL_UNORDERED_PAIR,
        assignment_entity="protein_group",
        projection_kind="ppi_c2_endpoint_pattern",
        default_grouping_kind="protein_identity",
        allowed_grouping_kinds=_PROTEIN_GROUPING_KINDS,
        projection_drop_reasons=(
            "c2_two_validation_endpoints",
            "c2_two_test_endpoints",
            "c2_validation_test_edge",
        ),
    ),
    (_PPI_TASK_ID, C3_SPLIT_STRATEGY): SplitStrategySpec(
        task_id=_PPI_TASK_ID,
        strategy_name=C3_SPLIT_STRATEGY,
        protocol_id="ppi.c3.v1",
        protocol_version=1,
        prediction_unit=_CANONICAL_UNORDERED_PAIR,
        assignment_entity="protein_group",
        projection_kind="ppi_c3_within_partition_edges",
        default_grouping_kind="protein_identity",
        allowed_grouping_kinds=_PROTEIN_GROUPING_KINDS,
        projection_drop_reasons=("c3_cross_partition_edge",),
    ),
}


def registered_split_strategies(task_id: str) -> tuple[str, ...]:
    """Return registered legacy strategy names for one task."""
    return tuple(sorted(
        strategy_name
        for registered_task, strategy_name in _SPLIT_STRATEGIES
        if registered_task == task_id
    ))


def get_split_strategy(task_id: str, strategy_name: str) -> SplitStrategySpec:
    """Return one compatible strategy or raise an actionable error."""
    key = (task_id, strategy_name)
    try:
        return _SPLIT_STRATEGIES[key]
    except KeyError:
        pass

    supported_tasks = sorted({
        registered_task
        for registered_task, registered_strategy in _SPLIT_STRATEGIES
        if registered_strategy == strategy_name
    })
    valid_strategies = registered_split_strategies(task_id)
    message = (
        f"Split strategy {strategy_name!r} is not supported for task "
        f"{task_id!r}."
    )
    if supported_tasks:
        message += (
            f" It is registered for task(s): {', '.join(supported_tasks)}."
        )
    if valid_strategies:
        message += (
            " Supported strategies for this task: "
            f"{', '.join(valid_strategies)}."
        )
    else:
        message += f" No split strategies are registered for task {task_id!r}."
    raise ValueError(message)
