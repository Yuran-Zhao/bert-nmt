# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import FairseqDataset
from typing import Optional


class TransformEosLangPairDataset(FairseqDataset):
    """A :class:`~fairseq.data.FairseqDataset` wrapper that transform bos on
    collated samples of language pair dataset.

    Note that the transformation is applied in :func:`collater`.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset that collates sample into
            LanguagePairDataset schema
        src_eos (int): original source end-of-sentence symbol index to be replaced
        new_src_eos (int, optional): new end-of-sentence symbol index to replace source eos symbol
        tgt_bos (int, optional): original target beginning-of-sentence symbol index to be replaced
        new_tgt_bos (int, optional): new beginning-of-sentence symbol index to replace at the
            beginning of 'prev_output_tokens'
    """
    def __init__(
        self,
        dataset: FairseqDataset,
        src_eos: int,
        new_src_eos: Optional[int] = None,
        tgt_bos: Optional[int] = None,
        new_tgt_bos: Optional[int] = None,
    ):
        self.dataset = dataset
        self.src_eos = src_eos
        self.new_src_eos = new_src_eos
        self.tgt_bos = tgt_bos
        self.new_tgt_bos = new_tgt_bos

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        samples = self.dataset.collater(samples)

        # TODO: support different padding direction
        if self.new_src_eos is not None:
            assert (samples['net_input']['src_tokens'][:, -1] !=
                    self.src_eos).sum() == 0
            samples['net_input']['src_tokens'][:, -1] = self.new_src_eos

        if self.new_tgt_bos is not None:
            assert (samples['net_input']['prev_output_tokens'][:, 0] !=
                    self.tgt_bos).sum() == 0
            samples['net_input']['prev_output_tokens'][:, 0] = self.new_tgt_bos

        return samples

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
