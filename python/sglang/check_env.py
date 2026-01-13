"""Check environment configurations and dependency versions."""

import importlib.metadata
import os
import resource
import subprocess
import sys
from abc import abstractmethod
from collections import OrderedDict, defaultdict

import torch


def is_cuda_v2():
    return torch.version.cuda is not None


# List of packages to check versions
PACKAGE_LIST = [
    "sglang",
    "sgl_kernel",
    "flashinfer_python",
    "flashinfer_cubin",
    "flashinfer_jit_cache",
    "triton",
    "transformers",
    "torchao",
    "numpy",
    "aiohttp",
    "fastapi",
    "hf_transfer",
    "huggingface_hub",
    "interegular",
    "modelscope",
    "orjson",
    "outlines",
    "packaging",
    "psutil",
    "pydantic",
    "python-multipart",
    "pyzmq",
    "torchao",
    "uvicorn",
    "uvloop",
    "vllm",
    "xgrammar",
    "openai",
    "tiktoken",
    "anthropic",
    "litellm",
    "decord2",
]


class BaseEnv:
    """Base class for environment check"""

    def __init__(self):
        self.package_list = PACKAGE_LIST

    @abstractmethod
    def get_info(self) -> dict:
        """
        Get CUDA-related information if available.
        """
        raise NotImplementedError

    @abstractmethod
    def get_topology(self) -> dict:
        raise NotImplementedError

    def get_package_versions(self) -> dict:
        """
        Get versions of specified packages.
        """
        versions = {}
        for package in self.package_list:
            package_name = package.split("==")[0].split(">=")[0].split("<=")[0]
            try:
                version = importlib.metadata.version(package_name)
                versions[package_name] = version
            except ModuleNotFoundError:
                versions[package_name] = "Module Not Found"
        return versions

    def get_device_info(self):
        """
        Get information about available GPU devices.
        """
        devices = defaultdict(list)
        capabilities = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
            capability = torch.cuda.get_device_capability(k)
            capabilities[f"{capability[0]}.{capability[1]}"].append(str(k))

        gpu_info = {}
        for name, device_ids in devices.items():
            gpu_info[f"GPU {','.join(device_ids)}"] = name

        if len(capabilities) == 1:
            # All GPUs have the same compute capability
            cap, gpu_ids = list(capabilities.items())[0]
            gpu_info[f"GPU {','.join(gpu_ids)} Compute Capability"] = cap
        else:
            # GPUs have different compute capabilities
            for cap, gpu_ids in capabilities.items():
                gpu_info[f"GPU {','.join(gpu_ids)} Compute Capability"] = cap

        return gpu_info

    def get_hypervisor_vendor(self) -> dict:
        try:
            output = subprocess.check_output(["lscpu"], text=True)
            for line in output.split("\n"):
                if "Hypervisor vendor:" in line:
                    return {"Hypervisor vendor:": line.split(":")[1].strip()}
            return {}
        except:
            return {}

    def get_ulimit_soft(self) -> dict:
        ulimit_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        return {"ulimit soft": ulimit_soft}

    def check_env(self):
        """
        Check and print environment information.
        """
        env_info = OrderedDict()
        env_info["Python"] = sys.version.replace("\n", "")
        env_info.update(self.get_info())
        env_info["PyTorch"] = torch.__version__
        env_info.update(self.get_package_versions())
        env_info.update(self.get_topology())
        env_info.update(self.get_hypervisor_vendor())
        env_info.update(self.get_ulimit_soft())

        for k, v in env_info.items():
            print(f"{k}: {v}")


class GPUEnv(BaseEnv):
    """Environment checker for Nvidia GPU"""

    def get_info(self):
        cuda_info = {"CUDA available": torch.cuda.is_available()}

        if cuda_info["CUDA available"]:
            cuda_info.update(self.get_device_info())
            cuda_info.update(self._get_cuda_version_info())

        return cuda_info

    def _get_cuda_version_info(self):
        """
        Get CUDA version information.
        """
        from torch.utils.cpp_extension import CUDA_HOME

        cuda_info = {"CUDA_HOME": CUDA_HOME}

        if CUDA_HOME and os.path.isdir(CUDA_HOME):
            cuda_info.update(self._get_nvcc_info())
            cuda_info.update(self._get_cuda_driver_version())

        return cuda_info

    def _get_nvcc_info(self):
        """
        Get NVCC version information.
        """
        from torch.utils.cpp_extension import CUDA_HOME

        try:
            nvcc = os.path.join(CUDA_HOME, "bin/nvcc")
            nvcc_output = (
                subprocess.check_output(f'"{nvcc}" -V', shell=True)
                .decode("utf-8")
                .strip()
            )
            return {
                "NVCC": nvcc_output[
                    nvcc_output.rfind("Cuda compilation tools") : nvcc_output.rfind(
                        "Build"
                    )
                ].strip()
            }
        except subprocess.SubprocessError:
            return {"NVCC": "Not Available"}

    def _get_cuda_driver_version(self):
        """
        Get CUDA driver version.
        """
        versions = set()
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ]
            )
            versions = set(output.decode().strip().split("\n"))
            if len(versions) == 1:
                return {"CUDA Driver Version": versions.pop()}
            else:
                return {"CUDA Driver Versions": ", ".join(sorted(versions))}
        except subprocess.SubprocessError:
            return {"CUDA Driver Version": "Not Available"}

    def get_topology(self):
        """
        Get GPU topology information.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return {
                "NVIDIA Topology": (
                    "\n" + result.stdout if result.returncode == 0 else None
                )
            }
        except subprocess.SubprocessError:
            return {}


if __name__ == "__main__":
    # NVIDIA CUDA only
    if is_cuda_v2():
        env = GPUEnv()
    else:
        raise RuntimeError("No supported GPU backend found (NVIDIA CUDA required)")
    env.check_env()
