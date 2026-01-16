# deploy.py

import os
import subprocess
from typing import Optional, List

from . import common
from . import Boxy
from . import Container
from . import Alloc 
from . import Vllm


class Deploy:
    """
    Orchestrates SLURM allocation, vLLM command construction, container image
    pulls/builds, and container launches—using injected collaborators.

    Args:
        system: name of the target system/cluster (sets CLUSTER env var)
        runtime: "podman" or "apptainer"
        alloc:    an Alloc-like instance (e.g. SallocModel)
        boxy:     a Boxy-like instance
        container: a Container-like instance (for pulling/building images)
        vllm:     a Vllm-like instance
    """

    def __init__(
        self,
        system: str,
        runtime: str,
        alloc: Alloc,
        boxy: Boxy,
        container: Container,
        vllm: Vllm,
    ):
        # 1) Export CLUSTER so Boxy picks it up
        os.environ["CLUSTER"] = system
        self.system = system

        # 2) Runtime selection
        runtime = runtime.lower()
        if runtime not in ("podman", "apptainer"):
            raise ValueError("runtime must be 'podman' or 'apptainer'")
        self.runtime = runtime

        # 3) Injected collaborators
        self.alloc = alloc
        self.boxy = boxy
        self.container = container
        self.vllm = vllm

        # 4) Internal state
        self.job_id: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None


    def allocate(self) -> str:
        """Run SLURM allocation if not already done."""
        if not self.job_id:
            self.job_id = self.alloc.allocate()
        return self.job_id


    def _build_container_cmd(self, extra_args: List[str]) -> List[str]:
        """Delegate to Boxy to assemble the podman/apptainer invocation."""
        if self.runtime == "podman":
            return self.boxy.build_podman_command(*extra_args)
        else:
            return self.boxy.build_apptainer_command(*extra_args)


    def run(self, background: bool = False) -> None:
        """
        Allocate nodes, ensure the container image is present, then
        launch the vLLM server inside a container.

        If background=True, returns immediately.  Otherwise, blocks
        until the process exits.
        """
        # 1) Allocate SLURM nodes
        #self.allocate()

        # 2) Optionally pull or build the container image
        image_ref = self.boxy.image_name if self.runtime == "podman" else f"{self.boxy.short_name}.sif"
        try:
            # only Podman makes sense for Container.pull
            if self.runtime == "podman":
                self.container.pull(image_ref)
        except Exception as e:
            # warn but continue
            print(f"Warning: failed to pull image '{image_ref}': {e}")

        # 3) Build the vLLM serve command
        full_vllm_cmd = self.vllm.build_command()
        # strip off the "vllm" entrypoint; Boxy will re-add it
        extra = full_vllm_cmd[1:]

        # 4) Build the container invocation
        cmd = self._build_container_cmd(extra)

        # 5) Launch
        if background:
            self._proc = subprocess.Popen(cmd)
        else:
            subprocess.run(cmd, check=True)


    def status(self) -> str:
        """
        Return status of the running container/vLLM:
          - "running" or "exited (code X)" if started in background
          - otherwise for Podman, query `podman ps`
        """
        if self._proc:
            code = self._proc.poll()
            return "running" if code is None else f"exited (code {code})"

        if self.runtime == "podman":
            name = self.boxy.container_name
            res = subprocess.run(
                ["podman", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Status}}"],
                capture_output=True, text=True
            )
            return res.stdout.strip() or "not running"

        return "no process handle; cannot determine status"


    def kill(self, release_nodes: bool = True) -> None:
        """
        Kill the container:
          - if launched in‐process, terminate that Popen
          - else if Podman, `podman kill <name>`
          - for Apptainer without a handle, raises
        Optionally releases the SLURM allocation.
        """
        # 1) kill the container
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
        elif self.runtime == "podman":
            subprocess.run(["podman", "kill", self.boxy.container_name], check=False)
        else:
            raise RuntimeError("No running Apptainer process handle available")

        # 2) release SLURM nodes
        if release_nodes and self.job_id:
            self.alloc.release()
            self.job_id = None
