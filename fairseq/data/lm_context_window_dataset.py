# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq.data.monolingual_dataset import MonolingualDataset

from . import FairseqDataset


class LMContextWindowDataset(FairseqDataset):
    """Wraps a MonolingualDataset and provides more context for evaluation."""
    def __init__(self, dataset, tokens_per_sample, context_window, pad_idx):
        assert isinstance(dataset, MonolingualDataset)
        assert context_window > 0
        self.dataset = dataset
        self.tokens_per_sample = tokens_per_sample
        self.context_window = context_window
        self.pad_idx = pad_idx
        self.prev_tokens = np.empty([0])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        sample = self.dataset.collater(samples)

        pad = self.pad_idx
        max_sample_len = self.tokens_per_sample + self.context_window

        bsz, tsz = sample['net_input']['src_tokens'].shape
        start_idxs = [0] * bsz
        toks = sample['net_input']['src_tokens']
        lengths = sample['net_input']['src_lengths']
        tgt = sample['target']
        new_toks = np.empty([bsz, tsz + self.context_window], dtype=np.int64)
        new_tgt = np.full([bsz, tsz + self.context_window], pad, dtype=np.int64)
        sample_lens = toks.ne(pad).long().sum(dim=1).cpu()
        for i in range(bsz):
            sample_len = sample_lens[i]
            extra = len(self.prev_tokens) + sample_len - max_sample_len
            if extra > 0:
                self.prev_tokens = self.prev_tokens[extra:]
            pads = np.full(self.context_window - len(self.prev_tokens), pad)
            new_toks[i] = np.concatenate(
                [self.prev_tokens, toks[i].numpy(), pads])
            new_tgt[i,
                    len(self.prev_tokens):len(self.prev_tokens) +
                    len(tgt[i])] = tgt[i]
            start_idxs[i] = len(self.prev_tokens)
            lengths[i] += len(self.prev_tokens)
            self.prev_tokens = new_toks[i][
                new_toks[i] != pad][-self.context_window:]
        sample['net_input']['src_tokens'] = torch.from_numpy(new_toks)
        sample['target'] = torch.from_numpy(new_tgt)
        sample['start_indices'] = start_idxs

        return sample

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        # NOTE we don't shuffle the data to retain access to the previous dataset elements
        return np.arange(len(self.dataset))

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
