# cosmos_predict1/utils/tensor_parallel_compat.py
import torch
from typing import Tuple

class VocabUtility:
    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Tuple[int, int]:
        # TP=1 => full vocab on rank 0
        if world_size <= 1:
            return 0, global_vocab_size
        # Simple contiguous partitioning (good enough if you ever enable TP>1)
        per = (global_vocab_size + world_size - 1) // world_size
        start = rank * per
        end = min(start + per, global_vocab_size)
        return start, end

def reduce_from_tensor_model_parallel_region(x: torch.Tensor) -> torch.Tensor:
    # TP=1 no-op
    return x

def reduce_scatter_to_sequence_parallel_region(x: torch.Tensor) -> torch.Tensor:
    # TP=1 no-op
    return x

class ColumnParallelLinear(torch.nn.Linear):
    """
    TP=1 fallback: behaves like nn.Linear.
    (Megatron returns (output, bias) sometimes; your wrapper expects that.)
    """
    def forward(self, input_: torch.Tensor):
        out = super().forward(input_)
        return out, None

class RowParallelLinear(torch.nn.Linear):
    def forward(self, input_: torch.Tensor):
        out = super().forward(input_)
        return out, None

class VocabParallelEmbedding(torch.nn.Embedding):
    pass


def gather_from_tensor_model_parallel_region(x: torch.Tensor) -> torch.Tensor:
    # TP=1 fallback: nothing to gather
    return x