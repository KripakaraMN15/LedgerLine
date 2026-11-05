from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from vendorops.telemetry import context_assemble_span


class ContextBudgetError(ValueError):
    """Raised when required prompt content cannot fit its assigned budget."""


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class ConservativeTokenCounter:
    chars_per_token: float = 4.0
    safety_margin: float = 1.1

    def __post_init__(self) -> None:
        if self.chars_per_token <= 0 or self.safety_margin < 1:
            raise ValueError("Token counter parameters must be conservative and positive")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) * self.safety_margin / self.chars_per_token)


@dataclass(frozen=True)
class ContextRatios:
    system: float = 0.15
    tools: float = 0.15
    documents: float = 0.40
    history: float = 0.30

    def __post_init__(self) -> None:
        values = (self.system, self.tools, self.documents, self.history)
        if any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError("Context ratios must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Context ratios must sum to 1")


@dataclass(frozen=True)
class RetrievedDocument:
    content: str
    score: float
    priority: int = 0
    document_id: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("Document score must be finite")


@dataclass(frozen=True)
class ContextReport:
    budget_tokens: int
    total_tokens: int
    system_tokens: int
    docs_tokens: int
    tools_tokens: int
    history_tokens: int
    dropped_docs: tuple[str, ...] = ()
    truncated_docs: tuple[str, ...] = ()
    truncated: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class AssembledContext:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    report: ContextReport


_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]*)?(?:\s+|$)")
_SECTIONS = ("system", "tools", "documents", "history")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _allocation(total: int, ratios: ContextRatios) -> dict[str, int]:
    exact = {
        "system": total * ratios.system,
        "tools": total * ratios.tools,
        "documents": total * ratios.documents,
        "history": total * ratios.history,
    }
    budgets = {section: math.floor(exact[section]) for section in _SECTIONS}
    remaining = total - sum(budgets.values())
    order = sorted(
        _SECTIONS,
        key=lambda section: exact[section] - budgets[section],
        reverse=True,
    )
    for section in order[:remaining]:
        budgets[section] += 1
    return budgets


def _message_tokens(counter: TokenCounter, messages: Sequence[dict[str, Any]]) -> int:
    if not messages:
        return 0
    return counter.count(_json_text(messages))


def _request_tokens(
    counter: TokenCounter,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
) -> int:
    return counter.count(_json_text({"messages": messages, "tools": tools}))


def _history_groups(history: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(history):
        message = history[index]
        group = [message]
        if message.get("role") == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                call_ids = {
                    call.get("id")
                    for call in calls
                    if isinstance(call, dict) and isinstance(call.get("id"), str)
                }
                next_index = index + 1
                while next_index < len(history):
                    result = history[next_index]
                    if (
                        result.get("role") != "tool"
                        or result.get("tool_call_id") not in call_ids
                    ):
                        break
                    group.append(result)
                    next_index += 1
                groups.append(group)
                index = next_index
                continue
        groups.append(group)
        index += 1
    return groups


def _latest_user_group(groups: Sequence[list[dict[str, Any]]]) -> int | None:
    latest: int | None = None
    for group_index, group in enumerate(groups):
        if any(message.get("role") == "user" for message in group):
            latest = group_index
    return latest


def _truncate_on_sentence(
    document: RetrievedDocument,
    *,
    selected_documents: Sequence[dict[str, Any]],
    document_budget: int,
    counter: TokenCounter,
    minimum_fraction: float,
) -> str | None:
    sentence_ends = [match.end() for match in _SENTENCE_END.finditer(document.content)]
    if not sentence_ends:
        return None

    low = 0
    high = len(sentence_ends)
    while low < high:
        middle = (low + high) // 2
        end = sentence_ends[middle]
        candidate = document.content[:end].rstrip()
        candidate_total = _message_tokens(
            counter,
            [
                *selected_documents,
                {"role": "system", "content": candidate},
            ],
        )
        if candidate_total <= document_budget:
            low = middle + 1
        else:
            high = middle
    if low == 0:
        return None

    best = document.content[: sentence_ends[low - 1]].rstrip()
    if counter.count(best) < counter.count(document.content) * minimum_fraction:
        return None
    return best


def assemble(
    *,
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
    documents: Sequence[RetrievedDocument] = (),
    history: Sequence[dict[str, Any]] = (),
    context_limit: int,
    reserved_output_tokens: int,
    ratios: ContextRatios | None = None,
    token_counter: TokenCounter | None = None,
    min_truncated_doc_fraction: float = 0.25,
) -> AssembledContext:
    if context_limit < 1 or reserved_output_tokens < 0:
        raise ValueError("Context and reserved-output limits must be non-negative")
    if reserved_output_tokens >= context_limit:
        raise ValueError("Reserved output tokens must be below the context limit")
    if not 0 < min_truncated_doc_fraction <= 1:
        raise ValueError("min_truncated_doc_fraction must be in (0, 1]")
    counter = token_counter or ConservativeTokenCounter()
    section_ratios = ratios or ContextRatios()
    budget_tokens = context_limit - reserved_output_tokens
    budgets = _allocation(budget_tokens, section_ratios)
    system_messages = [{"role": "system", "content": system_prompt}]
    system_tokens = _message_tokens(counter, system_messages)
    tools_tokens = counter.count(_json_text(tool_schemas)) if tool_schemas else 0
    if system_tokens > budgets["system"]:
        raise ContextBudgetError(
            f"System prompt requires {system_tokens} tokens; "
            f"its budget is {budgets['system']}"
        )
    if tools_tokens > budgets["tools"]:
        raise ContextBudgetError(
            f"Tool schemas require {tools_tokens} tokens; "
            f"their budget is {budgets['tools']}"
        )

    ranked_documents = sorted(
        enumerate(documents),
        key=lambda item: (-item[1].priority, -item[1].score, item[0]),
    )
    selected_docs: list[dict[str, Any]] = []
    selected_doc_ids: list[str] = []
    dropped_docs: list[str] = []
    truncated_docs: list[str] = []
    docs_tokens = 0
    for original_index, document in ranked_documents:
        identifier = document.document_id or str(original_index)
        full_message = {"role": "system", "content": document.content}
        full_total = _message_tokens(counter, [*selected_docs, full_message])
        if full_total <= budgets["documents"]:
            selected_docs.append(full_message)
            selected_doc_ids.append(identifier)
            docs_tokens = full_total
            continue

        truncated_content = _truncate_on_sentence(
            document,
            selected_documents=selected_docs,
            document_budget=budgets["documents"],
            counter=counter,
            minimum_fraction=min_truncated_doc_fraction,
        )
        if truncated_content is None:
            dropped_docs.append(identifier)
            continue
        truncated_message = {"role": "system", "content": truncated_content}
        truncated_total = _message_tokens(counter, [*selected_docs, truncated_message])
        if truncated_total > budgets["documents"]:
            dropped_docs.append(identifier)
            continue
        selected_docs.append(truncated_message)
        selected_doc_ids.append(identifier)
        docs_tokens = truncated_total
        truncated_docs.append(identifier)
        break

    groups = _history_groups(history)
    latest_user_group = _latest_user_group(groups)
    selected_group_indexes = set(range(len(groups)))
    selected_history = [message for group in groups for message in group]
    history_tokens = _message_tokens(counter, selected_history)
    for group_index in range(len(groups)):
        if history_tokens <= budgets["history"]:
            break
        if group_index == latest_user_group:
            continue
        selected_group_indexes.discard(group_index)
        selected_history = [
            message
            for index, messages in enumerate(groups)
            if index in selected_group_indexes
            for message in messages
        ]
        history_tokens = _message_tokens(counter, selected_history)

    messages = [*system_messages, *selected_docs, *selected_history]
    total_tokens = _request_tokens(counter, messages, tool_schemas)
    if history_tokens > budgets["history"]:
        raise ContextBudgetError(
            "Latest user message and indivisible history entries exceed the history budget"
        )

    for group_index in range(len(groups)):
        if total_tokens <= budget_tokens:
            break
        if (
            group_index == latest_user_group
            or group_index not in selected_group_indexes
        ):
            continue
        selected_group_indexes.discard(group_index)
        selected_history = [
            message
            for index, messages_in_group in enumerate(groups)
            if index in selected_group_indexes
            for message in messages_in_group
        ]
        history_tokens = _message_tokens(counter, selected_history)
        messages = [*system_messages, *selected_docs, *selected_history]
        total_tokens = _request_tokens(counter, messages, tool_schemas)

    while total_tokens > budget_tokens and selected_docs:
        selected_docs.pop()
        removed_id = selected_doc_ids.pop()
        dropped_docs.append(removed_id)
        if removed_id in truncated_docs:
            truncated_docs.remove(removed_id)
        docs_tokens = _message_tokens(counter, selected_docs)
        messages = [*system_messages, *selected_docs, *selected_history]
        total_tokens = _request_tokens(counter, messages, tool_schemas)

    if total_tokens > budget_tokens:
        raise ContextBudgetError(
            f"Assembled context requires {total_tokens} tokens; "
            f"available budget is {budget_tokens}"
        )
    report = ContextReport(
        budget_tokens=budget_tokens,
        total_tokens=total_tokens,
        system_tokens=system_tokens,
        docs_tokens=docs_tokens,
        tools_tokens=tools_tokens,
        history_tokens=history_tokens,
        dropped_docs=tuple(dropped_docs),
        truncated_docs=tuple(truncated_docs),
        truncated={
            "system": False,
            "tools": False,
            "documents": bool(truncated_docs),
            "history": len(selected_group_indexes) != len(groups),
        },
    )
    with context_assemble_span(
        budget_tokens=budget_tokens,
        system_tokens=system_tokens,
        docs_tokens=docs_tokens,
        tools_tokens=tools_tokens,
        dropped_docs=len(dropped_docs),
    ) as current_span:
        current_span.set_attribute("history_tokens", history_tokens)
        current_span.set_attribute("total_tokens", total_tokens)
        current_span.set_attribute("docs_truncated", bool(truncated_docs))
        current_span.set_attribute("history_truncated", report.truncated["history"])
    return AssembledContext(messages=messages, tools=tool_schemas, report=report)
