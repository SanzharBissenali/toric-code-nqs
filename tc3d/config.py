"""Runtime environment probe: device detection and chain-count defaults."""

import os
import platform
import subprocess
from typing import Any, Dict

import jax

# Speed-lever config keys late-bound from $TC3D_DEFAULTS_FILE -> their env-var
# names in that file (same three keys nersc/submit_nqs_gridinv.sh's own
# wrapper-level TC3D_DEFAULTS_FILE block reads).
_LEVER_ENV_KEYS = {"compute_dtype": "COMPUTE_DTYPE", "inv_impl": "INV_IMPL",
                   "qgt_solver": "QGT_SOLVER"}


def apply_late_lever_defaults(cfg: Dict[str, Any], *, prefix: str) -> Dict[str, Any]:
    """Late-bind compute_dtype/inv_impl/qgt_solver from $TC3D_DEFAULTS_FILE when
    the CLI gave no value at all -- the Python-side mirror of the wrapper's own
    TC3D_DEFAULTS_FILE block. Slurm copies the submitted BATCH SCRIPT at sbatch
    time, so an already-queued job runs the OLD wrapper text and never sees a
    wrapper-level fix added after it was queued; the Python package, unlike the
    script, is imported fresh at job start, so this is the actual safety net for
    those ~130 already-queued campaign jobs.

    First-start only: a no-op unless `RESUB_COUNT` is unset/"0" (a requeue's
    wrapper always re-passes the already-resolved env; late-binding a DIFFERENT
    default mid-chain, behind the requeue's back, is exactly what this must not
    do). Only fills a key that is truly unset (missing from `cfg`, i.e. neither
    an explicit flag nor an env-derived value reached it -- both `train.py` and
    `sweep.py` parse these with `default=argparse.SUPPRESS`, so "unset" means
    absent, and `dict.get` already reads that as None). Must run BEFORE
    `builders.with_defaults` (which would otherwise resolve an unset qgt_solver
    to "cholesky", masking "unset" as "explicitly cholesky") and before any
    resume-config check -- see `tc3d.io.check_resume_config`.

    Mutates and returns `cfg`. `TC3D_DEFAULTS_FILE=/dev/null` (or any missing
    path) disables this, same as the wrapper.
    """
    if os.environ.get("RESUB_COUNT", "0") != "0":
        return cfg
    path = os.environ.get("TC3D_DEFAULTS_FILE",
                          os.path.join(os.environ.get("PSCRATCH", "/nonexistent"),
                                       "tc_nqs", "phase3d", "defaults.env"))
    if not path or not os.path.isfile(path):
        return cfg
    file_vals: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            file_vals[k.strip()] = v.strip()
    for cfg_key, env_key in _LEVER_ENV_KEYS.items():
        if cfg.get(cfg_key) is not None:
            continue                                    # explicit flag/env wins
        val = file_vals.get(env_key, "")
        if val:
            cfg[cfg_key] = val
            print(f"[{prefix}] default from {path}: {cfg_key}={val}", flush=True)
    return cfg


def setup_environment():
    """Configure environment variables and print hardware information."""
    devices = jax.devices()
    has_gpu = any('cuda' in str(device).lower() for device in devices)

    if has_gpu:
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        gpu_assigned = "NVIDIA GPU"
        n_chains = 2**10  # 1024 chains for GPU

        try:
            command = 'nvidia-smi --query-gpu=gpu_name --format=csv,noheader'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            gpu_name, error = process.communicate()
            gpu_assigned = str(gpu_name)
        except FileNotFoundError:
            pass
    else:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        gpu_assigned = "CPU mode"
        n_chains = 2**4  # 16 chains for CPU

    try:
        process2 = subprocess.Popen(['hostname'], stdout=subprocess.PIPE)
        node_assigned, error = process2.communicate()
        node_assigned = str(node_assigned)
    except FileNotFoundError:
        node_assigned = platform.node()

    print("NODE:", node_assigned)
    print("ASSIGNED DEVICE:", gpu_assigned)
    print("NUMBER OF CHAINS:", n_chains)
    print("AVAILABLE DEVICES:", devices)

    return gpu_assigned, node_assigned, n_chains
