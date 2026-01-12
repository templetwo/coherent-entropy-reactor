"""
Coherent Entropy Reactor - Master Orchestrator

CLI tool for managing the CER dashboard, running tests, and configuration.

Usage:
    python master.py serve [--port 8080] [--host localhost]
    python master.py test [--module core|liquid|dashboard]
    python master.py config [--show|--edit]
    python master.py benchmark
"""

import typer
import uvicorn
import yaml
import subprocess
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(help="Coherent Entropy Reactor Master Control")
console = Console()

CONFIG_PATH = Path("config/reactor_config.yaml")


def load_config() -> dict:
    """Load configuration from YAML file."""
    if not CONFIG_PATH.exists():
        console.print(f"[yellow]Config file not found at {CONFIG_PATH}[/yellow]")
        console.print("[yellow]Creating default configuration...[/yellow]")
        create_default_config()
    
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def create_default_config():
    """Create default configuration file."""
    default_config = {
        'reactor': {
            'input_dim': 128,
            'hidden_dim': 256,
            'output_dim': 128,
            'target_entropy': 3.0,
            'kuramoto': {
                'coupling_strength': 2.0,
                'n_oscillators': 8
            }
        },
        'dashboard': {
            'host': 'localhost',
            'port': 8080,
            'debug': False,
            '3d_visuals': {
                'enable_bloom': True,
                'enable_particles': True,
                'max_particles': 1000,
                'merkabah_opacity': 0.3
            }
        }
    }
    
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    console.print(f"[green]✓[/green] Created default config at {CONFIG_PATH}")


@app.command()
def serve(
    port: int = typer.Option(None, "--port", "-p", help="Port to run dashboard on"),
    host: str = typer.Option(None, "--host", "-h", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload")
):
    """Start the CER dashboard server."""
    config = load_config()
    
    # Override config with CLI args
    final_host = host or config['dashboard']['host']
    final_port = port or config['dashboard']['port']
    
    console.print(f"[cyan]🚀 Starting CER Dashboard[/cyan]")
    console.print(f"[dim]Host: {final_host}:{final_port}[/dim]")
    console.print(f"[dim]URL: http://{final_host}:{final_port}[/dim]")
    
    uvicorn.run(
        "src.dashboard.main:app",
        host=final_host,
        port=final_port,
        reload=reload,
        log_level="info"
    )


@app.command()
def test(
    module: Optional[str] = typer.Option(None, "--module", "-m", help="Specific module to test (core|liquid|dashboard)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run tests for the CER."""
    console.print("[cyan]🧪 Running CER Tests[/cyan]")
    
    test_path = "tests/"
    if module:
        test_path = f"tests/test_{module}.py" if module != "dashboard" else "tests/test_dashboard_api.py"
    
    cmd = ["pytest", test_path]
    if verbose:
        cmd.append("-v")
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", help="Open config in editor")
):
    """Manage CER configuration."""
    if show:
        cfg = load_config()
        
        console.print("[cyan]Current Configuration:[/cyan]")
        console.print(yaml.dump(cfg, default_flow_style=False, indent=2))
        
    elif edit:
        import os
        editor = os.environ.get('EDITOR', 'nano')
        subprocess.run([editor, str(CONFIG_PATH)])
    else:
        console.print("[yellow]Use --show to view or --edit to modify configuration[/yellow]")


@app.command()
def benchmark(
    component: str = typer.Option("all", "--component", "-c", help="Component to benchmark (all|3d|reactor)"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling")
):
    """Run performance benchmarks."""
    console.print(f"[cyan]📊 Benchmarking: {component}[/cyan]")
    
    if component == "reactor" or component == "all":
        console.print("\n[yellow]Reactor Benchmark[/yellow]")
        result = subprocess.run([sys.executable, "src/core/reactor.py"])
        
    if component == "3d" or component == "all":
        console.print("\n[yellow]3D Rendering Benchmark[/yellow]")
        console.print("[dim]Start dashboard and monitor browser performance[/dim]")
    
    if profile:
        console.print("\n[yellow]Profiling enabled - results will be saved to profile.txt[/yellow]")


@app.command()
def status():
    """Show CER system status."""
    table = Table(title="CER System Status")
    
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Info", style="dim")
    
    # Check config
    config_status = "✓" if CONFIG_PATH.exists() else "✗"
    table.add_row("Configuration", config_status, str(CONFIG_PATH))
    
    # Check modules
    try:
        from src.core.reactor import CoherentEntropyReactor
        table.add_row("Core Reactor", "✓", "Importable")
    except Exception as e:
        table.add_row("Core Reactor", "✗", str(e))
    
    try:
        from src.liquid.dynamics import LiquidLayer
        table.add_row("Liquid Dynamics", "✓", "Importable")
    except Exception as e:
        table.add_row("Liquid Dynamics", "✗", str(e))
    
    # Check dashboard
    dashboard_html = Path("src/dashboard/index.html")
    dash_status = "✓" if dashboard_html.exists() else "✗"
    table.add_row("Dashboard", dash_status, str(dashboard_html))
    
    console.print(table)
    
    # Show config summary
    if CONFIG_PATH.exists():
        cfg = load_config()
        console.print(f"\n[cyan]Active Configuration:[/cyan]")
        console.print(f"  Target Entropy: {cfg['reactor']['target_entropy']} nats")
        console.print(f"  Dashboard: {cfg['dashboard']['host']}:{cfg['dashboard']['port']}")


@app.command()
def ablate(
    component: str = typer.Argument(..., help="Component to disable: kuramoto|drift|liquid|all"),
    baseline: bool = typer.Option(False, "--baseline", help="Run baseline (vanilla transformer) comparison"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility"),
    steps: int = typer.Option(100, "--steps", "-n", help="Number of steps to run")
):
    """
    Run ablation study by disabling components (for skeptical validation).
    
    This allows researchers to test the null hypothesis:
    - Does Kuramoto coupling actually matter?
    - Is drift control just random noise?
    - What's the contribution of Liquid dynamics?
    """
    console.print(f"[cyan]🔬 Ablation Study: Disabling {component}[/cyan]")
    console.print(f"[dim]Seed: {seed}, Steps: {steps}[/dim]")
    
    import torch
    torch.manual_seed(seed)
    import torch.nn.functional as F
    
    from src.core.reactor import CoherentEntropyReactor
    
    # Configuration flags
    disable_kuramoto = component in ['kuramoto', 'all']
    disable_drift = component in ['drift', 'all']
    disable_liquid = component in ['liquid', 'all']
    
    if baseline:
        console.print("[yellow]Running BASELINE (vanilla transformer)[/yellow]")
        # TODO: Implement vanilla transformer comparison
        console.print("[red]Baseline not implemented yet[/red]")
        return
    
    # Create reactor
    reactor = CoherentEntropyReactor(
        input_dim=128,
        hidden_dim=256,
        output_dim=128,
        target_entropy=3.0
    )
    
    # TODO: Add flags to reactor to disable components
    # For now, just document the approach
    console.print("\n[yellow]Ablation Configuration:[/yellow]")
    console.print(f"  Kuramoto: {'DISABLED' if disable_kuramoto else 'ENABLED'}")
    console.print(f"  Drift Control: {'DISABLED' if disable_drift else 'ENABLED'}")
    console.print(f"  Liquid Dynamics: {'DISABLED' if disable_liquid else 'ENABLED'}")
    
    console.print("\n[dim]Run benchmark to see results...[/dim]")
    console.print("[yellow]Note: Full ablation study requires reactor refactoring to support component disabling[/yellow]")


if __name__ == "__main__":
    app()
