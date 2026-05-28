import json
from pathlib import Path

import pytest

from multi_agent_lab.llm.json_extraction import JsonExtractionService
from multi_agent_lab.llm.metrics import LLMCallMetrics, LLMTraceRecorder
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient


def test_json_extraction_accepts_pure_json() -> None:
    extractor = JsonExtractionService()

    data = extractor.parse('{"content":"ok","reasoning_summary":"done"}', ["content"])

    assert data["content"] == "ok"


def test_json_extraction_accepts_markdown_json() -> None:
    extractor = JsonExtractionService()

    data = extractor.parse('```json\n{"content":"ok","reasoning_summary":"done"}\n```')

    assert data["reasoning_summary"] == "done"


def test_json_extraction_accepts_embedded_json() -> None:
    extractor = JsonExtractionService()

    data = extractor.parse('Here is the result: {"approved": true, "feedback": ""} thanks')

    assert data["approved"] is True


def test_json_extraction_repairs_common_errors() -> None:
    extractor = JsonExtractionService()

    data = extractor.parse("{'content':'ok','reasoning_summary':'done',}")

    assert data == {"content": "ok", "reasoning_summary": "done"}


def test_json_extraction_rejects_invalid_schema() -> None:
    extractor = JsonExtractionService()

    with pytest.raises(InvalidJSONError):
        extractor.parse('{"content":"ok"}', ["content", "reasoning_summary"])


async def test_generate_json_controlled_fallback_error() -> None:
    metrics = LLMCallMetrics()
    client = OllamaClient(
        use_mock=True,
        retries=0,
        mock_responses=["not-json"],  # type: ignore[list-item]
        metrics=metrics,
    )

    with pytest.raises(InvalidJSONError):
        await client.generate_json("CoderAgent", ["content"])

    assert metrics.failure_count == 1


async def test_generate_json_writes_traces(tmp_path: Path) -> None:
    metrics = LLMCallMetrics()
    trace_recorder = LLMTraceRecorder(tmp_path / "workspace" / ".traces")
    client = OllamaClient(
        use_mock=True,
        mock_responses=[{"content": "ok", "reasoning_summary": "done"}],
        metrics=metrics,
        trace_recorder=trace_recorder,
        agent_name="coder",
    )

    data = await client.generate_json("CoderAgent", ["content", "reasoning_summary"])

    trace_files = list((tmp_path / "workspace" / ".traces").glob("coder-*.json"))
    assert data["content"] == "ok"
    assert metrics.success_count == 1
    assert len(trace_files) == 1
    trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace["parsed_response"]["content"] == "ok"


async def test_ollama_request_uses_json_format_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    client = OllamaClient(use_mock=False)

    def fake_request(
        self: OllamaClient,
        method: str,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, str]:
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {"response": '{"content":"ok","reasoning_summary":"done"}'}

    monkeypatch.setattr(OllamaClient, "_request", fake_request)

    data = await client.generate_json("prompt", ["content", "reasoning_summary"])

    payload = captured["payload"]
    assert data["content"] == "ok"
    assert isinstance(payload, dict)
    assert payload["format"] == "json"
    assert payload["options"] == {"temperature": 0, "top_p": 0.9, "num_predict": 600}
