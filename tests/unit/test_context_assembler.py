from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from vendorops.agents.react import ReactAgent
from vendorops.context import (
    ConservativeTokenCounter,
    ContextBudgetError,
    ContextRatios,
    RetrievedDocument,
    assemble,
)
from vendorops.llm.client import LLMCompletion
from vendorops.models import Budget, TaskState, TaskStatus
from vendorops.telemetry import configure_tracing
from vendorops.tools import ToolDefinition, ToolRegistry, ToolRouter


@dataclass(frozen=True)
class ExactCharCounter:
    def count(self, text: str) -> int:
        return len(text)


@given(
    doc_lengths=st.lists(st.integers(min_value=0, max_value=300), min_size=0, max_size=5),
    history_lengths=st.lists(
        st.integers(min_value=0, max_value=200), min_size=0, max_size=12
    ),
    context_limit=st.integers(min_value=1024, max_value=8192),
)
@settings(max_examples=100, deadline=None)
def test_output_never_exceeds_available_context_budget(
    doc_lengths: list[int],
    history_lengths: list[int],
    context_limit: int,
) -> None:
    documents = [
        RetrievedDocument(
            content=("document. " * length),
            score=1.0 / (index + 1),
            document_id=str(index),
        )
        for index, length in enumerate(doc_lengths)
    ]
    history = [
        {"role": "user", "content": "latest " + ("x" * length)}
        for length in history_lengths
    ]
    counter = ConservativeTokenCounter()
    context = assemble(
        system_prompt="system",
        tool_schemas=[],
        documents=documents,
        history=history,
        context_limit=context_limit,
        reserved_output_tokens=128,
        token_counter=counter,
    )
    serialized_prompt = json.dumps(
        {"messages": context.messages, "tools": context.tools},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert counter.count(serialized_prompt) <= context_limit - 128
    assert context.report.total_tokens == counter.count(serialized_prompt)


def test_documents_are_ordered_by_priority_then_relevance() -> None:
    result = assemble(
        system_prompt="sys",
        tool_schemas=[],
        documents=[
            RetrievedDocument("lower priority high score.", score=1.0, priority=0),
            RetrievedDocument("highest priority low score.", score=0.1, priority=2),
            RetrievedDocument("same priority higher score.", score=0.9, priority=2),
        ],
        context_limit=3000,
        reserved_output_tokens=0,
        ratios=ContextRatios(system=0.1, tools=0.1, documents=0.7, history=0.1),
    )
    doc_contents = [message["content"] for message in result.messages[1:]]
    assert doc_contents == [
        "same priority higher score.",
        "highest priority low score.",
        "lower priority high score.",
    ]


def test_history_trimming_keeps_tool_call_and_results_together() -> None:
    tool_call = {
        "role": "assistant",
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "lookup"}}
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "result " * 200,
    }
    history = [
        {"role": "user", "content": "old " * 1000},
        tool_call,
        tool_result,
        {"role": "user", "content": "latest"},
    ]
    result = assemble(
        system_prompt="s",
        tool_schemas=[],
        history=history,
        context_limit=1200,
        reserved_output_tokens=0,
        ratios=ContextRatios(system=0.1, tools=0.1, documents=0.1, history=0.7),
        token_counter=ExactCharCounter(),
    )
    roles = [message.get("role") for message in result.messages]
    has_call = any(message.get("tool_calls") for message in result.messages)
    has_result = any(message.get("tool_call_id") == "call-1" for message in result.messages)
    assert roles[-1] == "user"
    assert not has_call and not has_result
    assert result.report.truncated["history"]


def test_history_keeps_a_complete_tool_pair_when_it_fits() -> None:
    call = {
        "role": "assistant",
        "tool_calls": [{"id": "pair", "type": "function", "function": {"name": "read"}}],
    }
    result_message = {"role": "tool", "tool_call_id": "pair", "content": "ok"}
    result = assemble(
        system_prompt="s",
        tool_schemas=[],
        history=[{"role": "user", "content": "old"}, call, result_message, {
            "role": "user",
            "content": "latest",
        }],
        context_limit=3000,
        reserved_output_tokens=0,
        ratios=ContextRatios(system=0.1, tools=0.1, documents=0.1, history=0.7),
        token_counter=ExactCharCounter(),
    )
    assert call in result.messages
    assert result_message in result.messages


def test_oversized_system_prompt_raises_instead_of_truncating() -> None:
    with pytest.raises(ContextBudgetError, match="System prompt"):
        assemble(
            system_prompt="system " * 100,
            tool_schemas=[],
            context_limit=200,
            reserved_output_tokens=0,
            ratios=ContextRatios(system=0.2, tools=0.2, documents=0.3, history=0.3),
            token_counter=ExactCharCounter(),
        )


def test_oversized_tool_schemas_raise_instead_of_truncating() -> None:
    with pytest.raises(ContextBudgetError, match="Tool schemas"):
        assemble(
            system_prompt="s",
            tool_schemas=[{"parameters": {"required": ["x" * 100]}}],
            context_limit=200,
            reserved_output_tokens=0,
            ratios=ContextRatios(system=0.2, tools=0.2, documents=0.3, history=0.3),
            token_counter=ExactCharCounter(),
        )


def test_last_document_is_truncated_at_a_sentence_boundary() -> None:
    first_sentence = "First sentence " + ("x" * 180) + "."
    second_sentence = "Second sentence " + ("y" * 180) + "."
    third_sentence = "Third sentence " + ("z" * 180) + "."
    result = assemble(
        system_prompt="sys",
        tool_schemas=[],
        documents=[
            RetrievedDocument(
                f"{first_sentence} {second_sentence} {third_sentence}",
                score=1.0,
                document_id="paragraph",
            )
        ],
        context_limit=500,
        reserved_output_tokens=0,
        ratios=ContextRatios(system=0.1, tools=0.05, documents=0.8, history=0.05),
        token_counter=ExactCharCounter(),
    )
    selected = result.messages[1]["content"]
    assert selected == first_sentence
    assert result.report.truncated_docs == ("paragraph",)


def test_empty_documents_and_history_are_supported() -> None:
    result = assemble(
        system_prompt="system",
        tool_schemas=[],
        documents=[],
        history=[],
        context_limit=1000,
        reserved_output_tokens=100,
    )
    assert result.messages == [{"role": "system", "content": "system"}]
    assert result.report.docs_tokens == 0
    assert result.report.history_tokens == 0
    assert result.report.dropped_docs == ()


def test_context_exactly_filling_budget_is_allowed() -> None:
    counter = ExactCharCounter()
    system_prompt = "sys"
    schemas = [{"name": "read"}]
    required = counter.count(
        '{"messages":[{"content":"sys","role":"system"}],"tools":[{"name":"read"}]}'
    )
    result = assemble(
        system_prompt=system_prompt,
        tool_schemas=schemas,
        context_limit=required,
        reserved_output_tokens=0,
        ratios=ContextRatios(
            system=0.55,
            tools=0.35,
            documents=0.05,
            history=0.05,
        ),
        token_counter=counter,
    )
    assert result.report.total_tokens == required
    assert result.report.total_tokens == result.report.budget_tokens


def test_pathological_huge_document_without_sentence_boundary_is_dropped() -> None:
    huge_document = "x" * 100_000
    result = assemble(
        system_prompt="sys",
        tool_schemas=[],
        documents=[
            RetrievedDocument(
                huge_document,
                score=1.0,
                document_id="unbroken",
            )
        ],
        context_limit=1000,
        reserved_output_tokens=100,
        ratios=ContextRatios(system=0.1, tools=0.1, documents=0.7, history=0.1),
    )
    assert result.report.dropped_docs == ("unbroken",)
    assert result.report.truncated_docs == ()
    assert result.report.total_tokens <= result.report.budget_tokens


def test_assembly_emits_required_context_span_attributes() -> None:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter, set_global=False)
    result = assemble(
        system_prompt="system",
        tool_schemas=[{"name": "read"}],
        documents=[RetrievedDocument("Doc sentence.", score=0.9)],
        history=[{"role": "user", "content": "hello"}],
        context_limit=2000,
        reserved_output_tokens=200,
    )
    span = next(span for span in exporter.get_finished_spans() if span.name == "context.assemble")
    assert span.attributes["budget_tokens"] == result.report.budget_tokens
    assert span.attributes["system_tokens"] == result.report.system_tokens
    assert span.attributes["docs_tokens"] == result.report.docs_tokens
    assert span.attributes["tools_tokens"] == result.report.tools_tokens
    assert span.attributes["dropped_docs"] == len(result.report.dropped_docs)


class _Input(BaseModel):
    key: str


class _Output(BaseModel):
    value: str


class _RecordingLLM:
    def __init__(self) -> None:
        self.responses = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "invalid",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"unexpected":"x"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "valid",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"key":"x"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"role": "assistant", "content": "Finished."}}]},
        ]
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        state: TaskState,
        messages: list[dict[str, Any]],
        *,
        model: str,
        headers: dict[str, str],
        tools: list[dict[str, Any]],
        estimated_prompt_tokens: int,
    ) -> LLMCompletion:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "estimated_prompt_tokens": estimated_prompt_tokens,
            }
        )
        return LLMCompletion(
            response=self.responses.pop(0),
            model=model,
            cost_usd=0.0,
            attempts=1,
        )


class _CheckpointRepository:
    async def checkpoint_task_state(
        self,
        task_id: str,
        state: dict[str, Any],
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_agent_assembles_initial_and_validation_retry_llm_calls() -> None:
    llm = _RecordingLLM()

    async def handler(args: dict[str, Any]) -> dict[str, str]:
        return {"value": args["key"]}

    registry = ToolRegistry()
    registry.register(ToolDefinition("lookup", "Look up a value.", _Input, _Output, handler))
    state = TaskState(
        task_id="task-1",
        status=TaskStatus.RUNNING,
        goal="Look up x",
        budget=Budget(),
        updated_at=datetime.now(UTC),
    )
    agent = ReactAgent(
        llm,  # type: ignore[arg-type]
        ToolRouter(registry, llm),  # type: ignore[arg-type]
        _CheckpointRepository(),
    )

    result = await agent.run(state)

    assert result.final_answer == "Finished."
    assert len(llm.calls) == 3
    for call in llm.calls:
        assert call["messages"][0]["role"] == "system"
        assert call["tools"]
        assert call["estimated_prompt_tokens"] > 0
    assert any(
        isinstance(message.get("content"), str)
        and "validation error" in message["content"].lower()
        for message in llm.calls[1]["messages"]
    )
