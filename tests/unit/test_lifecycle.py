import pytest

from vendorops.models import TaskStatus
from vendorops.orchestrator.lifecycle import (
    LEGAL_TRANSITIONS,
    InvalidTransition,
    validate_transition,
)

VALID_TRANSITIONS = sorted(
    [
        (source, target)
        for source, targets in LEGAL_TRANSITIONS.items()
        for target in targets
    ],
    key=lambda transition: (transition[0].value, transition[1].value),
)
INVALID_TRANSITIONS = [
    (source, target)
    for source in TaskStatus
    for target in TaskStatus
    if target not in LEGAL_TRANSITIONS[source]
]


@pytest.mark.parametrize(("source", "target"), VALID_TRANSITIONS)
def test_legal_transitions_are_accepted(source: TaskStatus, target: TaskStatus) -> None:
    validate_transition(source, target)


@pytest.mark.parametrize(("source", "target"), INVALID_TRANSITIONS)
def test_illegal_transitions_raise_invalid_transition(
    source: TaskStatus, target: TaskStatus
) -> None:
    with pytest.raises(InvalidTransition) as error:
        validate_transition(source, target)

    assert error.value.source is source
    assert error.value.target is target
