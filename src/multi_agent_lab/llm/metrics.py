"""LLM metrics and tracing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class LLMCallMetrics:
    """Aggregate LLM call metrics for one runtime."""

    success_count: int = 0
    failure_count: int = 0
    fallback_count: int = 0
    latencies: list[float] = field(default_factory=list)

    def record_success(self, latency_seconds: float) -> None:
        """Record a successful structured response."""
        self.success_count += 1
        self.latencies.append(latency_seconds)

    def record_failure(self, latency_seconds: float) -> None:
        """Record a failed structured response."""
        self.failure_count += 1
        self.latencies.append(latency_seconds)

    def record_fallback(self) -> None:
        """Record a deterministic fallback."""
        self.fallback_count += 1

    @property
    def average_latency(self) -> float:
        """Return average LLM latency in seconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def fallback_rate(self) -> float:
        """Return fallback percentage across calls."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.fallback_count / total


class LLMTraceRecorder:
    """Writes compact LLM traces under the configured workspace."""

    def __init__(self, trace_dir: str | Path | None = None) -> None:
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None

    def record(
        self,
        *,
        agent: str,
        model: str,
        prompt: str,
        raw_response: str,
        parsed_response: dict[str, Any] | None,
        fallback_reason: str | None,
        latency_seconds: float,
    ) -> None:
        """Persist a compact trace JSON file when tracing is enabled."""
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": agent,
            "model": model,
            "prompt_summary": prompt[:1000],
            "raw_response": raw_response[:4000],
            "parsed_response": parsed_response,
            "fallback_reason": fallback_reason,
            "latency_seconds": latency_seconds,
        }
        path = self.trace_dir / f"{agent}-{uuid4().hex}.json"
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


class LLMCallTimer:
    """Small context timer for LLM calls."""

    def __init__(self) -> None:
        self.started_at = perf_counter()

    def elapsed(self) -> float:
        """Return elapsed seconds."""
        return perf_counter() - self.started_at
