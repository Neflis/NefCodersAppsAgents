"""Small async Ollama client using the Python standard library."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from multi_agent_lab.llm.json_extraction import JsonExtractionService
from multi_agent_lab.llm.metrics import LLMCallMetrics, LLMCallTimer, LLMTraceRecorder
from multi_agent_lab.llm.ollama_client_types import InvalidJSONError, OllamaClientError


@dataclass(slots=True)
class OllamaClient:
    """Minimal client for the local Ollama HTTP API."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout: float = 10.0
    retries: int = 1
    use_mock: bool = True
    mock_responses: list[dict[str, Any]] = field(default_factory=list)
    metrics: LLMCallMetrics | None = None
    trace_recorder: LLMTraceRecorder | None = None
    agent_name: str = "unknown"
    extractor: JsonExtractionService = field(default_factory=JsonExtractionService)

    async def health_check(self) -> bool:
        """Return whether the Ollama API is reachable."""
        if self.use_mock:
            return True
        try:
            await asyncio.to_thread(self._request, "GET", "/api/tags")
        except OllamaClientError:
            return False
        return True

    async def generate(self, prompt: str) -> str:
        """Generate text with the configured Ollama model."""
        if self.use_mock:
            return json.dumps(self._next_mock_response(prompt))
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "top_p": 0.9, "num_predict": 600},
        }
        data = await asyncio.to_thread(self._request, "POST", "/api/generate", payload)
        response = data.get("response")
        if not isinstance(response, str):
            raise OllamaClientError("Ollama response did not include text.")
        return response

    async def generate_json(
        self,
        prompt: str,
        required_fields: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Generate and parse JSON with controlled retries."""
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            timer = LLMCallTimer()
            raw = ""
            parsed: dict[str, Any] | None = None
            fallback_reason: str | None = None
            try:
                raw = await self.generate(prompt)
                parsed = self._parse_with_schema(raw, required_fields)
                latency = timer.elapsed()
                if self.metrics is not None:
                    self.metrics.record_success(latency)
                self._trace(prompt, raw, parsed, None, latency)
                return parsed
            except (json.JSONDecodeError, InvalidJSONError, OllamaClientError) as error:
                last_error = error
                fallback_reason = str(error)
                latency = timer.elapsed()
                if self.metrics is not None:
                    self.metrics.record_failure(latency)
                self._trace(prompt, raw, parsed, fallback_reason, latency)
        raise InvalidJSONError(f"Invalid JSON from LLM: {last_error}") from last_error

    def record_fallback(self) -> None:
        """Record that caller used a deterministic fallback."""
        if self.metrics is not None:
            self.metrics.record_fallback()

    def _parse_with_schema(
        self,
        raw: str,
        required_fields: list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        """Parse current compact schemas while accepting legacy LLMDecision JSON."""
        try:
            return self.extractor.parse(raw, required_fields)
        except InvalidJSONError:
            parsed = self.extractor.parse(raw)
            if self._is_legacy_decision(parsed):
                return parsed
            raise

    def _is_legacy_decision(self, parsed: dict[str, Any]) -> bool:
        """Return whether parsed data matches the older LLMDecision shape."""
        has_reasoning = "reasoning_summary" in parsed
        has_content = "content" in parsed
        return has_reasoning and has_content

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute one JSON HTTP request against Ollama."""
        url = f"{self.base_url.rstrip('/')}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            raise OllamaClientError(f"Ollama HTTP error {error.code}.") from error
        except URLError as error:
            raise OllamaClientError(f"Ollama connection error: {error.reason}") from error
        except TimeoutError as error:
            raise OllamaClientError("Ollama request timed out.") from error

        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError as error:
            raise OllamaClientError("Ollama returned invalid JSON.") from error

    def _trace(
        self,
        prompt: str,
        raw_response: str,
        parsed_response: dict[str, Any] | None,
        fallback_reason: str | None,
        latency_seconds: float,
    ) -> None:
        """Write one LLM trace when configured."""
        if self.trace_recorder is None:
            return
        self.trace_recorder.record(
            agent=self.agent_name,
            model=self.model,
            prompt=prompt,
            raw_response=raw_response,
            parsed_response=parsed_response,
            fallback_reason=fallback_reason,
            latency_seconds=latency_seconds,
        )

    def _next_mock_response(self, prompt: str) -> dict[str, Any]:
        """Return a deterministic mock JSON response."""
        if self.mock_responses:
            return self.mock_responses.pop(0)
        if "PlannerAgent" in prompt:
            if "Flask" in prompt or "flask" in prompt:
                return {
                    "action": "decompose_goal",
                    "reasoning_summary": "Use deterministic Flask graph.",
                    "confidence": 1.0,
                    "events_to_publish": [],
                    "task_updates": [],
                    "content": None,
                }
            return {
                "action": "decompose_goal",
                "reasoning_summary": "Create the standard README task graph.",
                "confidence": 1.0,
                "events_to_publish": [],
                "task_updates": [],
                "content": [
                    {
                        "title": "Crear borrador README",
                        "description": "Generar contenido Markdown inicial.",
                        "required_capability": "coding",
                        "payload": {"path": "README.md"},
                        "priority": 10,
                    },
                    {
                        "title": "Revisar README",
                        "description": "Validar que el README propuesto sea aceptable.",
                        "required_capability": "reviewing",
                        "payload": {"path": "README.md"},
                        "priority": 8,
                    },
                    {
                        "title": "Escribir README en workspace",
                        "description": "Persistir el README aprobado.",
                        "required_capability": "file_write",
                        "payload": {"path": "README.md", "action": "write_file"},
                        "priority": 6,
                    },
                    {
                        "title": "Validar existencia del archivo",
                        "description": "Comprobar que el archivo existe.",
                        "required_capability": "testing_mock",
                        "payload": {"path": "README.md"},
                        "priority": 4,
                    },
                ],
            }
        if "CoderAgent" in prompt:
            target_path = self._mock_current_task_path(prompt)
            if target_path == "app.py":
                content = (
                    "from flask import Flask, jsonify, request\n\n"
                    "app = Flask(__name__)\n"
                    "todos = []\n\n"
                    "@app.get('/todos')\n"
                    "def list_todos():\n"
                    "    return jsonify(todos)\n\n"
                    "@app.post('/todos')\n"
                    "def create_todo():\n"
                    "    data = request.get_json(silent=True) or {}\n"
                    "    todo = {'id': len(todos) + 1, 'title': data.get('title', '')}\n"
                    "    todos.append(todo)\n"
                    "    return jsonify(todo), 201\n"
                )
            elif target_path == "requirements.txt":
                content = "Flask>=3.0\n"
            elif target_path == "tests/test_app.py":
                content = (
                    "from pathlib import Path\n\n\n"
                    "def test_project_files_are_coherent():\n"
                    "    app = Path('app.py').read_text(encoding='utf-8')\n"
                    "    requirements = Path('requirements.txt').read_text(encoding='utf-8')\n"
                    "    readme = Path('README.md').read_text(encoding='utf-8')\n"
                    "    assert 'from flask import' in app\n"
                    "    assert 'Flask' in requirements\n"
                    "    assert 'Flask TODO API' in readme\n"
                    "    assert '/todos' in readme\n"
                )
            elif target_path == "README.md" or "README.md" in prompt:
                content = (
                    "# Flask TODO API\n\n"
                    "Pequena API Flask para gestionar tareas TODO.\n\n"
                    "## Endpoints\n\n"
                    "- `GET /todos`\n- `POST /todos`\n"
                )
            else:
                content = "Estructura: app.py, requirements.txt y README.md " "para API Flask TODO."
            return {
                "action": "generate_content",
                "reasoning_summary": "Generate README content for the TODO app.",
                "confidence": 1.0,
                "events_to_publish": [],
                "task_updates": [],
                "content": content,
            }
        if "ReviewerAgent" in prompt:
            return {
                "action": "approve",
                "reasoning_summary": "Content is non-empty and safe for workspace writing.",
                "confidence": 1.0,
                "events_to_publish": [],
                "task_updates": [],
                "content": {"approved": True},
            }
        return {
            "action": "mock_decision",
            "reasoning_summary": "Deterministic mock response.",
            "confidence": 1.0,
            "events_to_publish": [],
            "task_updates": [],
            "content": {"prompt_excerpt": prompt[:120]},
        }

    def _mock_current_task_path(self, prompt: str) -> str | None:
        """Extract the active task path from a rendered mock prompt."""
        try:
            payload = json.loads(prompt.split("\n", 1)[1])
        except (IndexError, json.JSONDecodeError):
            return None

        context = payload.get("context", payload.get("input_context", {}))
        if not isinstance(context, dict):
            return None
        task = context.get("current_task", {})
        if not isinstance(task, dict):
            return None
        direct_path = task.get("path")
        if isinstance(direct_path, str):
            return direct_path
        task_payload = task.get("payload", {})
        if not isinstance(task_payload, dict):
            return None
        path = task_payload.get("path")
        return path if isinstance(path, str) else None
