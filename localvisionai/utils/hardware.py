"""Hardware detection and backend recommendation for LocalVisionAI."""

from __future__ import annotations

import os
import platform
import sys
from typing import Optional


def detect_hardware() -> dict:
    """
    Detect available compute hardware.

    Returns a dict with keys:
        platform, cuda, cuda_device_name, vram_gb, mps, mlx, cpu_cores, ram_gb
    """
    info: dict = {
        "platform": platform.system(),
        "cuda": False,
        "cuda_device_name": None,
        "vram_gb": None,
        "mps": False,    # Apple Silicon via PyTorch MPS
        "mlx": False,    # Apple Silicon via MLX
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": _get_ram_gb(),
    }

    # CUDA / MPS (via PyTorch)
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
        info["mps"] = torch.backends.mps.is_available()
    except ImportError:
        pass

    # MLX (Apple Silicon)
    try:
        import mlx.core  # noqa: F401
        info["mlx"] = True
    except ImportError:
        pass

    return info


def recommend_backend() -> str:
    """
    Suggest the best available backend for this machine.

    Priority: MLX (Apple Silicon) > Ollama (CUDA / CPU) > llama.cpp (CPU-only fallback)
    """
    hw = detect_hardware()
    if hw["mlx"]:
        return "mlx"
    if hw["cuda"]:
        # Ollama handles CUDA management automatically with minimal setup
        return "ollama"
    # Best CPU-only option: llama-cpp-python with GGUF quantized models
    return "llamacpp"


def _get_ram_gb() -> Optional[float]:
    """Return total system RAM in GB, or None if unavailable."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    # Fallback: read /proc/meminfo on Linux
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1e6, 1)
    except (OSError, ValueError):
        pass
    return None


def print_hardware_info() -> None:
    """Print a human-readable summary of detected hardware to stdout."""
    from rich.console import Console
    from rich.table import Table

    hw = detect_hardware()
    console = Console()

    table = Table(title="Hardware Detection", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Platform", hw["platform"])
    table.add_row("CPU Cores", str(hw["cpu_cores"]))
    table.add_row("RAM", f"{hw['ram_gb']} GB" if hw["ram_gb"] else "Unknown")
    table.add_row("CUDA Available", "✅" if hw["cuda"] else "❌")
    if hw["cuda"]:
        table.add_row("CUDA Device", hw.get("cuda_device_name", "Unknown"))
        table.add_row("VRAM", f"{hw['vram_gb']} GB" if hw["vram_gb"] else "Unknown")
    table.add_row("Apple MPS", "✅" if hw["mps"] else "❌")
    table.add_row("MLX (Apple Silicon)", "✅" if hw["mlx"] else "❌")
    table.add_row("Recommended Backend", f"[green]{recommend_backend()}[/green]")

    console.print(table)
