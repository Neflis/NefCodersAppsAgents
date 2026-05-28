"""Command-line interface for the multi-agent runtime."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from multi_agent_lab.runtime import AgentRuntime, RuntimeSummary

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Run the safe multi-agent runtime.")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run a goal through the agent network.")
    run_parser.add_argument("--goal", required=True, help="User goal to execute safely.")
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Use mock LLM decisions.")
    mode.add_argument("--ollama", action="store_true", help="Use real Ollama decisions.")
    run_parser.add_argument("--workspace", default="workspace", help="Workspace sandbox path.")
    run_parser.add_argument("--timeout", type=float, default=10.0, help="Global timeout seconds.")
    run_parser.add_argument("--verbose", action="store_true", help="Show detailed event logs.")
    run_parser.add_argument(
        "--allow-execution",
        action="store_true",
        help="Enable controlled command execution inside the workspace.",
    )

    parser.add_argument(
        "--mode",
        choices=["demo_mock", "demo_ollama"],
        help="Backward-compatible demo mode.",
    )
    return parser


async def run_demo(mode: str = "demo_mock") -> RuntimeSummary:
    """Run the default demo through AgentRuntime."""
    runtime = AgentRuntime(
        "Crear una pequena documentacion README para una app TODO",
        use_mock_llm=mode != "demo_ollama",
    )
    return await runtime.run()


async def run_from_args(args: argparse.Namespace) -> RuntimeSummary:
    """Run the runtime from parsed CLI arguments."""
    if args.command == "run":
        runtime = AgentRuntime(
            args.goal,
            workspace_path=args.workspace,
            use_mock_llm=not args.ollama,
            timeout_seconds=args.timeout,
            verbose=args.verbose,
            allow_execution=args.allow_execution,
        )
        return await runtime.run()

    return await run_demo(args.mode or "demo_mock")


def print_summary(summary: RuntimeSummary) -> None:
    """Print a human-readable runtime summary."""
    print(f"Objetivo: {summary.goal}")
    print(f"Estado final: {summary.status}")
    print(f"Tareas completadas: {summary.tasks_completed}")
    print(f"Tareas fallidas: {summary.tasks_failed}")
    print(f"Archivos creados: {', '.join(summary.files_created) or '(ninguno)'}")
    print(f"Duracion: {summary.duration_seconds:.2f}s")
    print(f"Correlation ID: {summary.correlation_id}")
    print(
        "LLM: "
        f"ok={summary.llm_success_count} "
        f"fail={summary.llm_failure_count} "
        f"fallbacks={summary.fallback_count} "
        f"avg_latency={summary.average_llm_latency:.2f}s"
    )
    print(
        "Execution: "
        f"ok={summary.execution_success_count} "
        f"fail={summary.execution_failure_count} "
        f"fix_attempts={summary.fix_attempts}"
    )
    if summary.final_failure_reason:
        print(f"Final failure: {summary.final_failure_reason}")
    event_counts = summary.event_summary.get("event_counts", {})
    if event_counts:
        print(f"Eventos: {event_counts}")


def main() -> None:
    """Run the command-line entrypoint."""
    logging.basicConfig(
        level=logging.INFO if "--verbose" in sys.argv else logging.WARNING,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    args = build_parser().parse_args()
    summary = asyncio.run(run_from_args(args))
    print_summary(summary)


if __name__ == "__main__":
    main()
