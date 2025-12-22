from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class InferenceParams:
    # Minimal fields your code uses
    max_batch_size: int
    max_sequence_length: int
    sequence_len_offset: int = 0

@dataclass
class ModelParallelConfig:
    # Minimal fields your code expects to exist
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1

class _ParallelStateStub:
    """Single-process / single-GPU stub that mimics megatron.core.parallel_state APIs used in inference."""

    def get_tensor_model_parallel_rank(self) -> int:
        return 0

    def get_tensor_model_parallel_world_size(self) -> int:
        return 1

    def get_context_parallel_world_size(self) -> int:
        return 1

    def get_context_parallel_rank(self) -> int:
        return 0

    def get_context_parallel_group(self) -> Any:
        # In single process, there is no process group.
        return None

    # Add any other getters your code calls, returning sane single-GPU defaults.

parallel_state = _ParallelStateStub()
