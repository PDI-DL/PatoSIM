#!/usr/bin/env python3
"""
sonar_viz.py - Visualizador de dados do sonar PatoSim.

Suporta: raw float32 .npy, PNG 16-bit e JPEG legado.
Nao depende de Isaac Sim.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def load_intensity(recording_dir: str, step: int) -> Optional[np.ndarray]:
    """Carrega a intensidade float32 de um step."""
    base = Path(recording_dir)

    npy = base / "state" / "sonar" / "raw" / f"{step:08d}.npy"
    if npy.exists():
        return np.load(str(npy)).astype(np.float32)

    png16 = base / "state" / "sonar" / "png16" / f"{step:08d}.png"
    if png16.exists():
        from PIL import Image

        img = Image.open(str(png16))
        return np.asarray(img, dtype=np.uint16).astype(np.float32) / 65535.0

    jpg = base / "state" / "rgb" / "sonar" / f"{step:08d}.jpg"
    if jpg.exists():
        from PIL import Image

        img = Image.open(str(jpg)).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0

    return None


def load_metadata(recording_dir: str, step: int) -> Optional[dict]:
    meta_path = Path(recording_dir) / "state" / "sonar" / "meta" / f"{step:08d}.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def list_available_steps(recording_dir: str) -> list[int]:
    """Retorna os steps com dados de sonar disponiveis."""
    base = Path(recording_dir)
    for subdir in ("state/sonar/raw", "state/sonar/png16", "state/rgb/sonar"):
        folder = base / subdir
        if folder.exists():
            files = sorted(folder.glob("*.npy")) + sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
            steps = []
            for path in files:
                try:
                    steps.append(int(path.stem))
                except ValueError:
                    pass
            if steps:
                return sorted(set(steps))
    return []


def project_to_fan(
    intensity: np.ndarray,
    metadata: Optional[dict] = None,
    output_size: int = 400,
) -> np.ndarray:
    """Reprojecao fan: grid polar (N_range x N_azi) -> imagem Cartesiana."""
    n_range, n_azi = intensity.shape
    if metadata and "sensor" in metadata:
        sensor = metadata["sensor"]
        min_r = float(sensor.get("min_range", 0.2))
        max_r = float(sensor.get("max_range", 10.0))
        hfov = float(sensor.get("hori_fov", 130.0))
    else:
        min_r, max_r, hfov = 0.2, 10.0, 130.0

    r_vals = np.linspace(min_r, max_r, n_range, dtype=np.float32)
    azi_deg_max = 90.0 + hfov / 2.0
    azi_deg_min = 90.0 - hfov / 2.0
    azi_vals = np.deg2rad(np.linspace(azi_deg_max, azi_deg_min, n_azi, dtype=np.float32))

    r_grid, azi_grid = np.meshgrid(r_vals, azi_vals, indexing="ij")
    x_cart = r_grid * np.cos(azi_grid)
    y_cart = r_grid * np.sin(azi_grid)

    lat_half = max_r * np.sin(np.deg2rad(hfov / 2.0))
    col_f = (x_cart + lat_half) / max(2.0 * lat_half, 1e-6) * (output_size - 1)
    row_f = (1.0 - np.clip(y_cart, 0, max_r) / max(max_r, 1e-6)) * (output_size - 1)
    col_i = np.clip(col_f.astype(np.int32), 0, output_size - 1)
    row_i = np.clip(row_f.astype(np.int32), 0, output_size - 1)

    out = np.zeros((output_size, output_size), dtype=np.float32)
    vals = intensity.ravel()
    rows = row_i.ravel()
    cols = col_i.ravel()
    np.maximum.at(out, (rows, cols), vals)
    return out


def plot_frame(
    recording_dir: str,
    step: int,
    *,
    show_hist: bool = False,
    save: bool = False,
    fan_size: int = 400,
) -> None:
    """Plota um frame: grid bruto + reprojecao fan + metadata."""
    intensity = load_intensity(recording_dir, step)
    if intensity is None:
        print(f"[sonar_viz] Nenhum dado encontrado para step {step}")
        sys.exit(1)

    meta = load_metadata(recording_dir, step)

    ncols = 3 if show_hist else 2
    fig = plt.figure(figsize=(5 * ncols, 5))
    gs = GridSpec(1, ncols, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(intensity, aspect="auto", cmap="gray", vmin=0, vmax=1, origin="upper")
    ax1.set_title(f"Grid r x azi (step {step:08d})")
    ax1.set_xlabel("Azimute bin")
    ax1.set_ylabel("Range bin")

    fan = project_to_fan(intensity, meta, fan_size)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(fan, aspect="equal", cmap="inferno", vmin=0, vmax=1, origin="upper")
    ax2.set_title("Fan polar (projecao Cartesiana)")
    ax2.set_xlabel("Lateral ->")
    ax2.set_ylabel("<- Frente")
    if meta and "sensor" in meta:
        sensor = meta["sensor"]
        ax2.set_title(
            "Fan polar\n"
            f"max_range={sensor.get('max_range', '?')} m  "
            f"hfov={sensor.get('hori_fov', '?')} deg"
        )

    if show_hist:
        ax3 = fig.add_subplot(gs[0, 2])
        flat = intensity.ravel()
        ax3.hist(flat[flat > 0.01], bins=128, color="steelblue", edgecolor="none")
        ax3.set_title("Histograma de intensidade")
        ax3.set_xlabel("Intensidade")
        ax3.set_ylabel("Contagem")
        ax3.set_xlim(0, 1)

    if meta:
        sensor = meta.get("sensor", {})
        render_params = meta.get("render_params", {})
        info = (
            f"N_range={sensor.get('n_range', '?')}  N_azi={sensor.get('n_azi', '?')}\n"
            f"range_res={sensor.get('range_res', '?')} m  "
            f"angular_res={sensor.get('angular_res', '?')} deg\n"
            f"atten={render_params.get('attenuation', '?')}  "
            f"gau={render_params.get('gau_noise_param', '?')}  "
            f"ray={render_params.get('ray_noise_param', '?')}"
        )
        fig.text(0.5, 0.01, info, ha="center", va="bottom", fontsize=8, family="monospace")

    plt.tight_layout()

    if save:
        out_path = os.path.join(recording_dir, f"sonar_viz_{step:08d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[sonar_viz] Salvo em {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_animation(
    recording_dir: str,
    start: int,
    end: int,
    fps: float = 10.0,
    fan_size: int = 300,
) -> None:
    """Cria animacao matplotlib dos frames start..end."""
    steps = [step for step in list_available_steps(recording_dir) if start <= step <= end]
    if not steps:
        print(f"[sonar_viz] Nenhum frame disponivel no range [{start}, {end}]")
        sys.exit(1)

    print(f"[sonar_viz] Animando {len(steps)} frames...")

    fig, (ax_grid, ax_fan) = plt.subplots(1, 2, figsize=(10, 5))
    first = load_intensity(recording_dir, steps[0])
    meta = load_metadata(recording_dir, steps[0])
    fan0 = project_to_fan(first, meta, fan_size)

    im_grid = ax_grid.imshow(first, aspect="auto", cmap="gray", vmin=0, vmax=1, origin="upper")
    im_fan = ax_fan.imshow(fan0, aspect="equal", cmap="inferno", vmin=0, vmax=1, origin="upper")
    ax_grid.set_title("Grid r x azi")
    ax_fan.set_title("Fan polar")
    title = fig.suptitle(f"step {steps[0]:08d}")

    def _update(frame_idx):
        step = steps[frame_idx]
        arr = load_intensity(recording_dir, step)
        if arr is None:
            return im_grid, im_fan
        fan = project_to_fan(arr, load_metadata(recording_dir, step), fan_size)
        im_grid.set_data(arr)
        im_fan.set_data(fan)
        title.set_text(f"step {step:08d}")
        return im_grid, im_fan

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(steps),
        interval=1000.0 / fps,
        blit=True,
    )
    _ = ani
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def print_summary(recording_dir: str) -> None:
    """Estatisticas gerais do dataset de sonar."""
    steps = list_available_steps(recording_dir)
    if not steps:
        print("[sonar_viz] Nenhum dado de sonar encontrado.")
        return
    print(f"[sonar_viz] Dataset: {recording_dir}")
    print(f"  Frames disponiveis: {len(steps)} ({steps[0]:08d} -> {steps[-1]:08d})")

    base = Path(recording_dir)
    for label, subdir in [
        ("raw .npy", "state/sonar/raw"),
        ("PNG 16-bit", "state/sonar/png16"),
        ("fan PNG", "state/sonar/polar_png"),
        ("meta JSON", "state/sonar/meta"),
        ("JPEG legado", "state/rgb/sonar"),
    ]:
        folder = base / subdir
        count = len(list(folder.glob("*.*"))) if folder.exists() else 0
        print(f"  {label:<14}: {count} arquivos")

    sample_steps = steps[:: max(1, len(steps) // 20)]
    means, stds = [], []
    for step in sample_steps:
        arr = load_intensity(recording_dir, step)
        if arr is not None:
            flat = arr.ravel()
            means.append(float(np.mean(flat)))
            stds.append(float(np.std(flat)))
    if means:
        print(f"  Intensidade media (amostra): {np.mean(means):.4f} +/- {np.mean(stds):.4f}")
        print(f"  Range: [{np.min(means):.4f}, {np.max(means):.4f}]")

    meta = load_metadata(recording_dir, steps[0])
    if meta and "sensor" in meta:
        sensor = meta["sensor"]
        print(
            f"  Sensor: N_range={sensor.get('n_range', '?')} "
            f"N_azi={sensor.get('n_azi', '?')} "
            f"max_range={sensor.get('max_range', '?')}m "
            f"hfov={sensor.get('hori_fov', '?')}deg"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizador de dados do sonar PatoSim (sem dependencia de Isaac Sim)"
    )
    parser.add_argument("--data", required=True, help="Diretorio do recording PatoSim")
    parser.add_argument("--frame", type=int, help="Step para visualizar")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=999999)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--fan-size", type=int, default=400)
    parser.add_argument("--hist", action="store_true", help="Mostrar histograma")
    parser.add_argument("--save", action="store_true", help="Salvar PNG em vez de exibir")
    parser.add_argument("--summary", action="store_true", help="Resumo do dataset")
    parser.add_argument("--animate", action="store_true", help="Animacao de frames")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.data)
    elif args.animate:
        plot_animation(args.data, args.start, args.end, args.fps, args.fan_size)
    elif args.frame is not None:
        plot_frame(
            args.data,
            args.frame,
            show_hist=args.hist,
            save=args.save,
            fan_size=args.fan_size,
        )
    else:
        steps = list_available_steps(args.data)
        if steps:
            plot_frame(args.data, steps[0], show_hist=args.hist, save=args.save)
        else:
            print("[sonar_viz] Nenhum dado encontrado.")
            sys.exit(1)


if __name__ == "__main__":
    main()
