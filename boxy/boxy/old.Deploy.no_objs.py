# deploy.py

import os
import subprocess
from typing import Optional, List

import common
from boxy import Boxy
from salloc_model import SallocModel as Alloc   # your Alloc wrapper around salloc
from vllm import Vllm


class Deploy:
    """
    Orchestrates:
      1) SLURM allocation via Alloc
      2) vLLM command construction via Vllm
      3) Container invocation via Boxy

    User can specify:
      - system (aka CLUSTER)
      - model_name (must be in common.models)
      - container runtime ("podman" or "apptainer")

    You can then:
      - run the container in foreground or background
      - kill it
      - query its status
    """

    def __init__(
        self,
        system          : str,
        model_name      : str,
        runtime         : str = "podman",
        alloc_nodes     : int = 1,
        alloc_time      : str = "01:00:00",
        alloc_partition : Optional[str] = None,
        # Any kwargs you want to pass on to Vllm
        vllm_kwargs     : Optional[dict] = None,
    ):
        # 1) Set CLUSTER for Boxy to pick up
        os.environ["CLUSTER"] = system
        self.system = system

        # 2) Instantiate the SLURM allocator
        self.alloc = Alloc(
            nodes=alloc_nodes,
            time=alloc_time,
            partition=alloc_partition,
        )
        self.job_id: Optional[str] = None

        # 3) Validate model_name and build the vLLM command
        self.vllm = Vllm(
            model_name=model_name,
            **(vllm_kwargs or {}),
        )

        # 4) Instantiate Boxy for the chosen runtime
        self.runtime = runtime.lower()
        if self.runtime not in ("podman", "apptainer"):
            raise ValueError("runtime must be 'podman' or 'apptainer'")
        self.boxy = Boxy()

        # 5) Will hold subprocess.Popen if run in background
        self._proc: Optional[subprocess.Popen] = None

    def allocate(self) -> str:
        """Run salloc to get nodes. Returns the SLURM job ID."""
        if self.job_id is None:
            self.job_id = self.alloc.allocate()
        return self.job_id

    def _build_container_cmd(self, extra_args: List[str]) -> List[str]:
        """
        Given the vLLM args (everything after the 'vllm' entrypoint),
        build the full container invocation.
        """
        if self.runtime == "podman":
            return self.boxy.build_podman_command(*extra_args)
        else:
            return self.boxy.build_apptainer_command(*extra_args)

    def run(self, background: bool = False) -> None:
        """
        Allocate nodes (if not already), then launch the container
        running `vllm serve ...` inside it.

        If background=True, returns immediately and you can
        later call `.status()` or `.kill()`. Otherwise blocks
        until the container (and vLLM) exits.
        """
        # 1) allocate nodes
        self.allocate()

        # 2) build the vLLM serve command
        vllm_cmd = self.vllm.build_command()
        # strip off the first element ("vllm") because Boxy
        # already sets entrypoint for us
        extra = vllm_cmd[1:]

        # 3) build container invocation
        cmd = self._build_container_cmd(extra)

        # 4) launch
        if background:
            # leave it running in the background
            self._proc = subprocess.Popen(cmd)
        else:
            # run in foreground, block until it finishes
            subprocess.run(cmd, check=True)

    def status(self) -> str:
        """
        Returns a simple status string:
          - For background runs: whether the Popen is still alive or exited.
          - Otherwise (or if no Popen): for Podman, query `podman ps`.
        """
        if self._proc:
            if self._proc.poll() is None:
                return "running"
            else:
                return f"exited (code {self._proc.poll()})"

        if self.runtime == "podman":
            name = self.boxy.container_name
            result = subprocess.run(
                ["podman", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
            )
            out = result.stdout.strip()
            return out or "not running"

        # For Apptainer with no Popen handle, we have no global registry
        return "no process handle; cannot determine status"

    def kill(self, release_nodes: bool = True) -> None:
        """
        Kill the running container:
          - If started in background via Popen, terminate that process.
          - Otherwise for Podman, use `podman kill <name>`.

        If release_nodes=True, also call `scancel` on the SLURM job.
        """
        # 1) kill the container invocation
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
        elif self.runtime == "podman":
            name = self.boxy.container_name
            subprocess.run(["podman", "kill", name], check=False)
        else:
            raise RuntimeError("No running container process to kill")

        # 2) optionally release the SLURM allocation
        if release_nodes and self.job_id:
            self.alloc.release()
            self.job_id = None
