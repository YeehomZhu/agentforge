"""CLI entry — wraps the compiled graph behind a simple `ask` command."""
from __future__ import annotations

import typer
from langchain_core.messages import HumanMessage
from rich.console import Console

from agentforge.graph.build import build_graph
from agentforge.tracing import new_conversation_id

app = typer.Typer(help="AgentForge — hierarchical multi-agent reference impl.")
console = Console()


@app.command()
def ask(question: str) -> None:
    cid = new_conversation_id()
    graph = build_graph()
    result = graph.invoke(
        {
            "conversation_id": cid,
            "messages": [HumanMessage(content=question)],
        }
    )
    console.rule(f"conversation_id: {cid}")
    console.print(result.get("final_answer", "[no answer produced]"))
    if result.get("fallback_triggered"):
        console.print(f"[yellow]fallback: {result.get('fallback_reason')}[/yellow]")
    if score := result.get("critic_score"):
        console.print(f"[dim]critic.aggregate={score['aggregate']:.2f}[/dim]")


if __name__ == "__main__":
    app()
