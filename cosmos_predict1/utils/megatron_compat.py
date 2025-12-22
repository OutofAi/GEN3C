# cosmos_predict1/utils/megatron_compat.py
from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceParams:
    max_batch_size: int
    max_sequence_length: int
    sequence_len_offset: int = 0

@dataclass
class ModelParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1

class _ParallelStateStub:
    def get_tensor_model_parallel_rank(self) -> int: return 0
    def get_tensor_model_parallel_world_size(self) -> int: return 1
    def get_tensor_model_parallel_group(self) -> Any: return None

    def get_context_parallel_rank(self) -> int: return 0
    def get_context_parallel_world_size(self) -> int: return 1
    def get_context_parallel_group(self) -> Any: return None

parallel_state = _ParallelStateStub()

class _MPUStub:
    def get_context_parallel_rank(self) -> int:
        return 0

mpu = _MPUStub()
