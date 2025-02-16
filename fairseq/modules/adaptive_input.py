# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn

from typing import List


class AdaptiveInput(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        initial_dim: int,
        factor: float,
        output_dim: int,
        cutoff: List[int],
    ):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[
                -1], 'cannot specify cutoff larger than vocab size'

        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx

        self.embeddings = nn.ModuleList()
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // (factor**i))
            seq = nn.Sequential(nn.Embedding(size, dim, padding_idx),
                                nn.Linear(dim, output_dim, bias=False))
            self.embeddings.append(seq)

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1]**-0.5)
                nn.init.constant_(m.weight[padding_idx], 0)
            elif hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def weights_for_band(self, band: int):
        return self.embeddings[band][0].weight, self.embeddings[band][1].weight

    def forward(self, input: torch.Tensor):
        result = self._float_tensor.new(input.shape + (self.embedding_dim, ))
        for i in range(len(self.cutoff)):
            mask = input.lt(self.cutoff[i])
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))
                chunk_input = input[mask] - self.cutoff[i - 1]
            else:
                chunk_input = input[mask]
            if mask.any():
                result[mask] = self.embeddings[i](chunk_input)
        return result
