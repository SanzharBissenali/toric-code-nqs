"""Runtime environment probe: device detection and chain-count defaults."""

import os
import platform
import subprocess

import jax


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
