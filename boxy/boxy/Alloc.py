import subprocess



class SallocModel:
    """
    A simple model for allocating (and releasing) SLURM compute nodes via salloc.
    
    Example:
        alloc = SallocModel(nodes=2, time="00:30:00", partition="debug")
        job_id = alloc.allocate()
        # ... run work under that allocation ...
        alloc.release()
    """

    def __init__(self, nodes=1, time="01:00:00", partition=None, name=None,
                 cluster=cluster, gpus_per_node=None):
        self._cluster = cluster
        self._nodes = nodes
        self._time = time
        self._partition = partition
        self._job_id = None
        self._job_name = name
        self._gpus_per_node = gpus_per_node

        self._cmd = None

        
    def allocate(self):
        """
        Call `salloc` (with --parsable) to reserve nodes.
        Returns the job ID string on success.
        Raises RuntimeError on failure.
        """
        cmd = [
            "salloc",
            f"--nodes={self._nodes}",
            f"--time={self._time}",
            "--parsable",
        ]

        if self._partition:
            cmd += [ f"--partition={self._partition}" ]
        if self._job_name:
            cmd += [ f"--job-name={self._job_name}" ]
        if self._gpus_per_node:
            cmd += [ f"--gpus-per-node={self._gpus_per_node}" ]


        self._cmd = cmd

        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            self._job_id = out.strip()
            return self._job_id
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"salloc failed (exit {e.returncode}):\n{e.output}"
            ) from e

        
    def get_job_id(self):
        """Return the current SLURM job ID, or None if no allocation yet."""
        return self._job_id

    
    def get_cmd(self):
        """Return the command as an array of strings."""
        return self._cmd

    
    def release(self):
        """
        Cancel the SLURM job (releasing the nodes).
        Raises RuntimeError if no job is allocated.
        """
        if not self._job_id:
            raise RuntimeError("No SLURM job allocated to release")

        try:
            subprocess.run(["scancel", self._job_id], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"scancel failed (exit {e.returncode})") from e
        finally:
            self._job_id = None
