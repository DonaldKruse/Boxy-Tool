# boxy.py

import os
import shutil
import subprocess
from typing import List


class Boxy:
    """
    Prototype “boxy” deployer for vLLM containers on Podman or Apptainer.

    Usage:
        box = Boxy()
        box.check_podman()
        cmd = box.build_podman_command("--port", "8000")
        subprocess.run(cmd)

    Or for Apptainer:
        box = Boxy()
        box.check_apptainer()
        box.build_apptainer_image()
        cmd = box.build_apptainer_command("--port", "8000")
        subprocess.run(cmd)
    """

    def __init__(self):
        # General Configuration
        self.registry = os.environ.get("REGISTRY", "")
        # PLATFORM alias: same as CLUSTER env var in your Bash
        self.platform = os.environ.get("CLUSTER", "unknown")
        self.cluster = os.environ.get("CLUSTER", "")

        # vLLM Configuration
        self.host_models_path = "./models"
        self._select_image_and_target()

        self.container_name = "vllm"
        self.short_name = f"vllm-{self.target}"
        self.work_dir = "/vllm-workspace/models"
        self.container_entrypoint = "vllm"

        # Base environment variables for the container
        self.env_vars: List[str] = [
            "OMP_NUM_THREADS=1",
            "HF_HUB_ENABLE_HF_TRANSFER=0",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "VLLM_NO_USAGE_STATS=1",
            "DO_NOT_TRACK=1",
            "HF_DATASETS_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "HF_HUB_OFFLINE=1",
            "VLLM_DISABLE_COMPILE_CACHE=1",
            "VLLM_ENABLE_V1_MULTIPROCESSING=0",
        ]
        if self.target == "rocm":
            # ROCm-specific env
            self.env_vars += [
                "VLLM_USE_V1=1",
                "VLLM_USE_TRITON_FLASH_ATTN=0",
            ]

        # Podman arguments
        self.podman_args: List[str] = [
            "--rm",
            f"--name={self.container_name}",
            "--network=host",
            "--ipc=host",
            f"--entrypoint={self.container_entrypoint}",
            f"--workdir={self.work_dir}",
            f"--volume={self.host_models_path}:{self.work_dir}",
        ]
        # add GPU support flags
        if self.target == "cuda":
            self.podman_args.append("--device=nvidia.com/gpu=all")
        else:  # rocm
            self.podman_args += [
                "--group-add=video",
                "--cap-add=SYS_PTRACE",
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--security-opt=seccomp=unconfined",
            ]

        # Apptainer arguments
        self.apptainer_args: List[str] = [
            "--fakeroot",
            "--writable-tmpfs",
            "--cleanenv",
            "--no-home",
            f"--cwd {self.work_dir}",
            f"--bind {self.host_models_path}:{self.work_dir}",
            "--env HF_HOME=/root/.cache/huggingface",
        ]
        # add GPU support flags
        if self.target == "cuda":
            self.apptainer_args.append("--nv")
        else:
            self.apptainer_args.append("--rocm")

        # Arguments tacked on at the very end
        self.container_args: List[str] = []
        if self.cluster == "eldorado":
            self.container_args.append("--gpu-memory-utilization=0.7")
        # make vLLM deterministic
        self.container_args.append("--seed=12345")

    def _select_image_and_target(self) -> None:
        """
        Pick IMAGE_NAME and TARGET based on self.platform / self.cluster.
        """
        if self.platform in ("hops", "unknown"):
            self.image_name = f"{self.registry}vllm/vllm-openai:v0.9.1"
            self.target = "cuda"
        elif self.cluster == "eldorado":
            self.image_name = (
                f"{self.registry}rocm/vllm:rocm6.4.1_vllm_0.9.1_20250702"
            )
            self.target = "rocm"
        else:
            raise RuntimeError(
                f"Unsupported platform/cluster '{self.platform}' / '{self.cluster}'"
            )

    def check_podman(self) -> None:
        """
        Ensure `podman` is on PATH and clear XDG vars if in a Slurm/Flux job.
        """
        if shutil.which("podman") is None:
            raise RuntimeError("Podman not found in PATH. Please install Podman.")
        # clear XDG if in an interactive job
        if os.environ.get("SLURM_JOB_ID") or os.environ.get("FLUX_ENCLOSING_ID"):
            os.environ.pop("XDG_SESSION_ID", None)
            os.environ.pop("XDG_RUNTIME_DIR", None)

    def build_podman_command(self, *extra_args: str) -> List[str]:
        """
        Assemble the full `podman run ...` command.
        extra_args are placed after the image name and before container_args.
        """
        cmd: List[str] = ["podman", "run"] + self.podman_args

        # add env vars
        for ev in self.env_vars:
            cmd += ["--env", ev]

        # image
        cmd.append(self.image_name)

        # user‐supplied extra args
        cmd += list(extra_args)

        # default container args (seed, gpu‐util, etc)
        cmd += self.container_args

        return cmd

    def check_apptainer(self) -> None:
        """
        Ensure `apptainer` is on PATH and (for ROCm) load rocm module if missing.
        """
        if shutil.which("apptainer") is None:
            raise RuntimeError("Apptainer not found in PATH. Please install Apptainer.")

        if self.target == "rocm":
            # try to detect a loaded rocm module
            res = subprocess.run(
                "module list 2>&1", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if "rocm" not in res.stdout:
                # attempt to load it
                subprocess.run("module load rocm/6.4.0", shell=True, check=True)

    def build_apptainer_image(self) -> None:
        """
        If the .sif doesn't exist, build it via `apptainer build`.
        """
        sif = f"{self.short_name}.sif"
        if not os.path.isfile(sif):
            # mimic your Bash APPTAINER_CACHEDIR export
            os.environ["APPTAINER_CACHEDIR"] = "./apptainer_cachedir"
            subprocess.run(
                ["apptainer", "build", "--force", sif, f"docker://{self.image_name}"],
                check=True,
            )

    def build_apptainer_command(self, *extra_args: str) -> List[str]:
        """
        Assemble the full `apptainer exec ...` command.
        extra_args are placed after the entrypoint and before container_args.
        """
        cmd: List[str] = ["apptainer", "exec"] + self.apptainer_args

        # add env vars
        for ev in self.env_vars:
            cmd += ["--env", ev]

        # point to the SIF
        cmd.append(f"{self.short_name}.sif")

        # entrypoint inside the container
        cmd.append(self.container_entrypoint)

        # user‐supplied extra args
        cmd += list(extra_args)

        # default container args
        cmd += self.container_args

        return cmd
