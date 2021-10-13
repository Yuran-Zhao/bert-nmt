"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open
import json
from transformers import BartTokenizer as Tokenizer

from .file_utils import cached_path

logger = logging.getLogger(__name__)


def load_vocab(vocab_file):
    """Loads a vocabulary json file into a dictionary."""
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    return collections.OrderedDict(vocab)


class BartTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""
    def __init__(
        self,
        vocab_file,
        # do_lower_case=True,
        max_len=None,
        #  do_basic_tokenize=True,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BartTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        # if not os.path.isfile(vocab_file):
        #     raise ValueError(
        #         "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
        #         "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
        #         .format(vocab_file))
        self.tokenizer = Tokenizer.from_pretrained('facebook/bart-base',
                                                   cache_dir='bart-base')
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        # self.do_basic_tokenize = do_basic_tokenize
        # if do_basic_tokenize:
        #     self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                           never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
        self.unk_word = "<unk>"
        self.unk_index = self.vocab[self.unk_word]
        self.pad_word = "<pad>"
        self.pad_index = self.vocab[self.pad_word]
        self.cls_word = "<s>"
        self.cls_index = self.vocab[self.cls_word]
        self.sep_word = "</s>"
        self.sep_index = self.vocab[self.sep_word]

    def tokenize(self, text):
        output_ids = self.tokenizer.encode(text)[1:-1]
        return [self.ids_to_tokens[idx] for idx in output_ids]
        # split_tokens = []
        # if self.do_basic_tokenize:
        #     for token in self.basic_tokenizer.tokenize(text):
        #         for sub_token in self.wordpiece_tokenizer.tokenize(token):
        #             split_tokens.append(sub_token)
        # else:
        #     split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # return split_tokens

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.vocab)

    def pad(self):
        return self.pad_index

    def cls(self):
        return self.cls_index

    def sep(self):
        return self.sep_index

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(
                    len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        # index = 0
        # if os.path.isdir(vocab_path):
        #     vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        # with open(vocab_file, "w", encoding="utf-8") as writer:
        #     for token, token_index in sorted(self.vocab.items(),
        #                                      key=lambda kv: kv[1]):
        #         if index != token_index:
        #             logger.warning(
        #                 "Saving vocabulary to {}: vocabulary indices are not consecutive."
        #                 " Please check that the vocabulary is not corrupted!".
        #                 format(vocab_file))
        #             index = token_index
        #         writer.write(token + u'\n')
        #         index += 1
        # return vocab_file
        pass

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        cache_dir=None,
                        *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        # if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
        #     vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[
        #         pretrained_model_name_or_path]
        #     if '-cased' in pretrained_model_name_or_path and kwargs.get(
        #             'do_lower_case', True):
        #         logger.warning(
        #             "The pre-trained model you are loading is a cased model but you have not set "
        #             "`do_lower_case` to False. We are setting `do_lower_case=False` for you but "
        #             "you may want to check this behavior.")
        #         kwargs['do_lower_case'] = False
        #     elif '-cased' not in pretrained_model_name_or_path and not kwargs.get(
        #             'do_lower_case', True):
        #         logger.warning(
        #             "The pre-trained model you are loading is an uncased model but you have set "
        #             "`do_lower_case` to False. We are setting `do_lower_case=True` for you "
        #             "but you may want to check this behavior.")
        #         kwargs['do_lower_case'] = True
        # else:
        #     vocab_file = pretrained_model_name_or_path
        # if os.path.isdir(vocab_file):
        #     vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        # # redirect to the cache, if necessary
        # try:
        #     resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        # except EnvironmentError:
        #     logger.error(
        #         "Model name '{}' was not found in model name list ({}). "
        #         "We assumed '{}' was a path or url but couldn't find any file "
        #         "associated to this path or url.".format(
        #             pretrained_model_name_or_path,
        #             ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()), vocab_file))
        #     return None
        # if resolved_vocab_file == vocab_file:
        #     logger.info("loading vocabulary file {}".format(vocab_file))
        # else:
        #     logger.info("loading vocabulary file {} from cache at {}".format(
        #         vocab_file, resolved_vocab_file))
        # if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
        #     # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
        #     # than the number of positional embeddings
        #     max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[
        #         pretrained_model_name_or_path]
        #     kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # # Instantiate tokenizer.
        tokenizer = cls(pretrained_model_name_or_path,
                        cache_dir=cache_dir,
                        *inputs,
                        **kwargs)
        return tokenizer