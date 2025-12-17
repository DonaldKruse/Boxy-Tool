# vllm.py

import os
from typing import List, Optional
import common

class Vllm:
    """
    Constructs a `vllm serve` command for a model defined in common.models.

    Args:
        model_name: must be one of common.models
        tensor_parallel_size: value for --tensor-parallel-size
        max_model_len: value for --max_model_len
        override_generation_config: path or JSON string for --override-generateion-config
        args: any additional CLI tokens to append
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 2048,
        override_generation_config: Optional[str] = '{\"attn_temperature_tuning\": true}',
        args: Optional[List[str]] = None,
    ):
        if model_name not in common.models:
            raise ValueError(f"Model '{model_name}' is not in common.models: {common.models!r}")
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.override_generation_config = override_generation_config
        self.args = args or []
        self.cmd = None

    def build_command(self) -> List[str]:
        """
        Returns the full `vllm serve` command as a list of strings.
        """
        model_path = os.path.join(common.MODEL_DIR, self.model_name)

        cmd: List[str] = [
            "vllm",
            "serve",
            model_path,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--disable-log-requests",
            "--max_model_len", str(self.max_model_len),
        ]

        if self.override_generation_config:
            cmd.extend([
                "--override-generateion-config",
                self.override_generation_config,
            ])

        # append any extra positional or flag args
        if self.args:
            cmd.extend(self.args)

        self.cmd = cmd
        return cmd
