"""Eval harness — run a suite, emit metrics, compare against baseline.

Usage:
    python -m agentforge.eval.run --suite eval/suites/baseline.yaml
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@dataclass
class CaseResult:
    case_id: str
    factuality: float
    tool_select: float
    latency_seconds: float
    fallback_triggered: bool
    notes: str = ""


@dataclass
class SuiteResult:
    suite_name: str
    cases: list[CaseResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = len(self.cases) or 1
        latencies = sorted(c.latency_seconds for c in self.cases)
        return {
            "n": len(self.cases),
            "factuality": sum(c.factuality for c in self.cases) / n,
            "tool_select": sum(c.tool_select for c in self.cases) / n,
            "latency_p50": latencies[int(0.50 * (n - 1))] if latencies else 0.0,
            "latency_p95": latencies[int(0.95 * (n - 1))] if latencies else 0.0,
            "fallback_rate": sum(1 for c in self.cases if c.fallback_triggered) / n,
        }


@app.command()
def run(
    suite: Path = typer.Option(..., exists=True, help="Path to eval suite YAML"),
    judges: str = typer.Option("llm", help="Comma-separated judges: llm, human-sample"),
) -> None:
    suite_def = yaml.safe_load(suite.read_text())
    cases = list(_load_cases(suite_def))

    result = SuiteResult(suite_name=suite_def["suite"])
    for case in cases:
        result.cases.append(_run_case(case, judges=judges.split(",")))

    _render(result)


def _load_cases(suite_def: dict) -> list[dict]:
    # TODO: glob cases from suite_def["cases"] path
    raise NotImplementedError


def _run_case(case: dict, judges: list[str]) -> CaseResult:
    # TODO:
    # 1. invoke graph with case["input"]
    # 2. score with each judge
    # 3. record latency
    t0 = time.perf_counter()
    # ... invoke graph ...
    elapsed = time.perf_counter() - t0
    return CaseResult(
        case_id=case["id"],
        factuality=0.0,
        tool_select=0.0,
        latency_seconds=elapsed,
        fallback_triggered=False,
    )


def _render(result: SuiteResult) -> None:
    s = result.summary()
    table = Table(title=f"Eval: {result.suite_name}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("cases", str(s["n"]))
    table.add_row("factuality", f"{s['factuality']:.2f}")
    table.add_row("tool_select", f"{s['tool_select']:.2f}")
    table.add_row("latency_p50", f"{s['latency_p50']:.2f}s")
    table.add_row("latency_p95", f"{s['latency_p95']:.2f}s")
    table.add_row("fallback_rate", f"{s['fallback_rate']:.2f}")
    console.print(table)


if __name__ == "__main__":
    app()
