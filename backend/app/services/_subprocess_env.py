"""Build subprocess env that exposes conda's CUDA libs via LD_LIBRARY_PATH.

Why this is needed:
  Modern CUDA-bound libraries (bitsandbytes, vllm's compiled kernels, etc.)
  call into shared objects like `libnvJitLink.so.13` via `ctypes.CDLL`.
  ctypes resolves library paths *at process startup* using LD_LIBRARY_PATH;
  modifying `os.environ["LD_LIBRARY_PATH"]` inside a running Python process
  does NOT reach the already-initialized dynamic linker. The fix has to be
  applied before the child Python interpreter is spawned, i.e. in the
  parent's subprocess.Popen(env=...) argument.

  When you `conda install -c nvidia cuda-toolkit=13.0`, the SOs land under
  `<env_root>/lib/` (and sometimes `<env_root>/targets/x86_64-linux/lib/`)
  but conda activation only adds `<env_root>/bin` to PATH — the lib dir
  is NOT exported to LD_LIBRARY_PATH automatically. Hence this helper.
"""
import os
import sys
from pathlib import Path


def _conda_lib_dirs() -> list[str]:
    """Candidate lib directories under the active Python's conda env, filtered to dirs that exist."""
    # sys.executable is e.g. /home/x/miniconda3/envs/cdss/bin/python →
    # env root is the parent of `bin`.
    env_root = Path(sys.executable).resolve().parent.parent
    candidates = [
        env_root / "lib",                                         # most conda nvidia packages
        env_root / "lib64",
        env_root / "targets" / "x86_64-linux" / "lib",            # newer cuda-runtime layout
        # `nvidia-*` PyPI wheels (when installed instead of conda packages)
        # ship shared objects into site-packages/nvidia/<lib>/lib/. We add the
        # most common ones; missing dirs are filtered below.
        env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
                  / "site-packages" / "nvidia" / "cuda_runtime" / "lib",
        env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
                  / "site-packages" / "nvidia" / "nvjitlink" / "lib",
        env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
                  / "site-packages" / "nvidia" / "cublas" / "lib",
        env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
                  / "site-packages" / "nvidia" / "cudnn" / "lib",
    ]
    return [str(p) for p in candidates if p.is_dir()]


def build_subprocess_env(extra: dict | None = None) -> dict:
    """Return an env dict for subprocess.Popen with conda CUDA libs prepended to LD_LIBRARY_PATH.

    `extra` lets the caller add (and override) keys like PYTHONUNBUFFERED, CUDA_VISIBLE_DEVICES.
    Existing LD_LIBRARY_PATH entries from the user's shell are preserved at the back.
    """
    env = {**os.environ}
    cuda_libs = _conda_lib_dirs()
    if cuda_libs:
        existing = env.get("LD_LIBRARY_PATH", "")
        new_paths = ":".join(cuda_libs)
        env["LD_LIBRARY_PATH"] = (
            f"{new_paths}:{existing}" if existing else new_paths
        )

    # Suggest PyTorch's expandable allocator so long-running fine-tunes
    # don't accumulate fragmentation that turns into spurious "tried to
    # allocate N GiB" OOMs late in a run. Don't override if the user
    # already set their own value via shell env.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # On hosts with mixed-model GPUs (e.g. V100-DGXS + V100-PCIE on the same
    # box) the default CUDA enumeration is FASTEST_FIRST, which can reorder
    # device indices in a way that doesn't match `nvidia-smi`. Forcing
    # PCI_BUS_ID guarantees CUDA_VISIBLE_DEVICES=N picks the same physical
    # card the user saw in nvidia-smi.
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    if extra:
        env.update(extra)
    return env
