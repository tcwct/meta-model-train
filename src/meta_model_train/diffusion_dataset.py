"""
2D periodic diffusion: spectral trajectory generation and online k-step batch helpers.

Physics (L, D, T, nt, nx, ny) come from Diffusion2DConfig. Metadata fields u0_source,
u0_seed, and cfg_seed describe the initial condition; cfg.seed is only the default
sampling seed when u0 is generated inside the function and u0_seed is omitted.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

try:
    import h5py
except ImportError:  # optional for training-only workflows
    h5py = None  # type: ignore[assignment]
import numpy as np

DataMode = Literal["full", "slices"]


@dataclass(frozen=True)
class Diffusion2DConfig:
    nx: int = 16
    ny: int = 16
    L: float = 1.0
    D: float = 0.005
    T: float = 5.0
    nt: int = 501  # includes t=0 and t=T
    seed: int = 42


def make_periodic_grid(L: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    dx = L / nx
    dy = L / ny
    x = np.linspace(0.0, L, nx, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, L, ny, endpoint=False, dtype=np.float64)
    return x, y, dx, dy


def sample_u0_uniform(nx: int, ny: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(ny, nx)).astype(np.float64)


def _build_k2(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    Kx, Ky = np.meshgrid(kx, ky)
    return (Kx**2 + Ky**2).astype(np.float64)


@lru_cache(maxsize=32)
def _cached_k2_and_t(
    nx: int,
    ny: int,
    L: float,
    nt: int,
    T: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """K2 (ny,nx) float64 and t (nt,) float64; depends only on grid and time discretization."""
    _x, _y, dx, dy = make_periodic_grid(L, nx, ny)
    K2 = _build_k2(nx, ny, dx=dx, dy=dy)
    t = np.linspace(0.0, T, nt, dtype=np.float64)
    return K2, t


@lru_cache(maxsize=128)
def _cached_single_step_decays(
    nx: int,
    ny: int,
    L: float,
    nt: int,
    T: float,
    D: float,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decay factors for u_k and u_{k+1}; depends only on physics config and k."""
    K2, t = _cached_k2_and_t(nx, ny, L, nt, T)
    decay_k = np.exp(-D * K2 * t[k]).astype(np.complex128)
    decay_k1 = np.exp(-D * K2 * t[k + 1]).astype(np.complex128)
    return decay_k, decay_k1


def generate_diffusion_2d_trajectory(
    cfg: Diffusion2DConfig,
    u0: Optional[np.ndarray] = None,
    u0_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate u(t,y,x) for 2D diffusion equation with periodic boundary using spectral method.

    If u0 is None, the initial field is sampled with seed (u0_seed if not None else cfg.seed).
    If u0 is provided, u0_seed is optional metadata only (recorded in attrs when set).
    """
    x, y, dx, dy = make_periodic_grid(cfg.L, cfg.nx, cfg.ny)
    K2, t = _cached_k2_and_t(cfg.nx, cfg.ny, float(cfg.L), int(cfg.nt), float(cfg.T))

    if u0 is None:
        effective_u0_seed = int(cfg.seed if u0_seed is None else u0_seed)
        u0 = sample_u0_uniform(cfg.nx, cfg.ny, effective_u0_seed)
        u0_source = "sampled"
    else:
        u0 = np.asarray(u0, dtype=np.float64)
        if u0.shape != (cfg.ny, cfg.nx):
            raise ValueError(f"u0 shape must be (ny,nx)=({cfg.ny},{cfg.nx}), got {u0.shape}")
        u0_source = "external"
        effective_u0_seed = int(u0_seed) if u0_seed is not None else None

    u_hat0 = np.fft.fft2(u0)

    u = np.empty((cfg.nt, cfg.ny, cfg.nx), dtype=np.float32)
    u[0] = u0.astype(np.float32)
    for i in range(1, cfg.nt):
        decay = np.exp(-cfg.D * K2 * t[i])
        ui = np.real(np.fft.ifft2(u_hat0 * decay))
        u[i] = ui.astype(np.float32)

    attrs: Dict[str, Any] = {
        "equation": "diffusion_2d",
        "method": "spectral",
        "periodic": True,
        "nx": int(cfg.nx),
        "ny": int(cfg.ny),
        "L": float(cfg.L),
        "D": float(cfg.D),
        "T": float(cfg.T),
        "nt": int(cfg.nt),
        "dx": float(dx),
        "dy": float(dy),
        "u0_distribution": "uniform_0_1_iid",
        "u0_source": u0_source,
        "cfg_seed": int(cfg.seed),
    }
    if u0_source == "sampled":
        attrs["u0_seed"] = int(effective_u0_seed)
        attrs["seed"] = int(cfg.seed)
    elif effective_u0_seed is not None:
        attrs["u0_seed"] = int(effective_u0_seed)

    return u, t.astype(np.float32), x.astype(np.float32), y.astype(np.float32), attrs


def extract_k_step_frames(u: np.ndarray, k: int, frame_stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float32)
    if u.ndim != 3:
        raise ValueError(f"u must be 3D (nt,ny,nx), got shape {u.shape}")
    nt = u.shape[0]
    s = int(frame_stride)
    if s < 1:
        raise ValueError(f"frame_stride must be >= 1, got {s}")
    max_k = nt - 1 - 2 * s
    if k < 0 or k > max_k:
        raise ValueError(
            f"k must satisfy 0 <= k <= nt-1-2*frame_stride ({max_k} for nt={nt}, frame_stride={s}), got k={k}"
        )
    inputs = np.stack((u[k], u[k + s]), axis=0)
    target = u[k + 2 * s : k + 2 * s + 1]
    return inputs, target


def extract_single_step_frames(u: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float32)
    if u.ndim != 3:
        raise ValueError(f"u must be 3D (nt,ny,nx), got shape {u.shape}")
    nt = u.shape[0]
    max_k = nt - 2
    if k < 0 or k > max_k:
        raise ValueError(f"k must satisfy 0 <= k <= nt-2 ({max_k} for nt={nt}), got k={k}")
    inputs = u[k : k + 1]
    target = u[k + 1 : k + 2]
    return inputs, target


def sample_u0_seeds_batch(base_seed: int, step: int, batch_size: int) -> np.ndarray:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    i = np.arange(batch_size, dtype=np.int64)
    return base_seed + int(step) * int(batch_size) + i


def _normalize_seeds_1d(u0_seeds: Union[np.ndarray, Sequence[int]]) -> np.ndarray:
    seeds = np.asarray(u0_seeds, dtype=np.int64)
    if seeds.ndim != 1:
        raise ValueError(f"u0_seeds must be 1-D, got shape {seeds.shape}")
    if int(seeds.shape[0]) < 1:
        raise ValueError("u0_seeds must be non-empty")
    return seeds


def _u_at_time_index(
    cfg: Diffusion2DConfig,
    u0: np.ndarray,
    u_hat0: np.ndarray,
    K2: np.ndarray,
    t: np.ndarray,
    time_index: int,
) -> np.ndarray:
    if time_index == 0:
        return u0.astype(np.float32)
    decay = np.exp(-cfg.D * K2 * t[time_index])
    return np.real(np.fft.ifft2(u_hat0 * decay)).astype(np.float32)


def generate_single_step_batch(
    cfg: Diffusion2DConfig,
    k: int,
    u0_seeds: Union[np.ndarray, Sequence[int]],
    *,
    data_mode: DataMode = "slices",
) -> Tuple[np.ndarray, np.ndarray]:
    seeds = _normalize_seeds_1d(u0_seeds)
    B = int(seeds.shape[0])
    inputs = np.empty((B, 1, cfg.ny, cfg.nx), dtype=np.float32)
    target = np.empty((B, 1, cfg.ny, cfg.nx), dtype=np.float32)

    if k < 0 or k > cfg.nt - 2:
        raise ValueError(f"k must satisfy 0 <= k <= nt-2 ({cfg.nt - 2}), got {k}")

    if data_mode == "full":
        for i in range(B):
            seed = int(seeds[i])
            u0 = sample_u0_uniform(cfg.nx, cfg.ny, seed)
            u = generate_diffusion_2d_trajectory(cfg, u0=u0, u0_seed=seed)[0]
            inp, tgt = extract_single_step_frames(u, k)
            inputs[i] = inp
            target[i] = tgt
        return inputs, target

    if data_mode != "slices":
        raise ValueError(f"unsupported data_mode={data_mode}")

    decay_k, decay_k1 = _cached_single_step_decays(
        int(cfg.nx),
        int(cfg.ny),
        float(cfg.L),
        int(cfg.nt),
        float(cfg.T),
        float(cfg.D),
        int(k),
    )
    u0_batch = np.stack([sample_u0_uniform(cfg.nx, cfg.ny, int(seed)) for seed in seeds], axis=0)
    u_hat0_batch = np.fft.fft2(u0_batch, axes=(-2, -1))
    u_k = np.fft.ifft2(u_hat0_batch * decay_k[None, :, :], axes=(-2, -1)).real.astype(np.float32)
    u_k1 = np.fft.ifft2(u_hat0_batch * decay_k1[None, :, :], axes=(-2, -1)).real.astype(np.float32)
    inputs[:, 0] = u_k
    target[:, 0] = u_k1
    return inputs, target
