import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.file_utils import cached_path
from transformers.models.bart.configuration_bart import BartConfig

from transformers.modeling_bart import (PretrainedBartModel, LayerNorm,
                                        EncoderLayer, DecoderLayer,
                                        LearnedPositionalEmbedding,
                                        _prepare_bart_decoder_inputs,
                                        _make_linear_from_emb)
import pdb

from sem_extractor import SemExtractor


class SemExtractorEncoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.dropout = config.dropout
        self.embed_tokens = embed_tokens

        self.embed_synt = nn.Embedding(77, config.d_model, config.pad_token_id)
        self.embed_synt.weight.data.normal_(mean=0.0, std=config.init_std)
        self.embed_synt.weight.data[config.pad_token_id].zero_()

        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id,
            config.extra_pos_embeddings)

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.synt_layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.syntax_encoder_layer_num)
        ])

        self.layernorm_embedding = LayerNorm(config.d_model)

        self.synt_layernorm_embedding = LayerNorm(config.d_model)

        self.pooling = MeanPooling(config)
        # self.pooling = AttentionPooling(config)

    def forward(self, input_ids, attention_mask):

        input_token_ids, input_synt_ids = torch.split(
            input_ids,
            [self.config.max_sent_len + 2, self.config.max_synt_len + 2],
            dim=1)
        input_token_mask, input_synt_mask = torch.split(
            attention_mask,
            [self.config.max_sent_len + 2, self.config.max_synt_len + 2],
            dim=1)

        x = self.forward_token(input_token_ids, input_token_mask)
        y = self.forward_synt(input_synt_ids, input_synt_mask)

        encoder_outputs = torch.cat((x, y), dim=1)

        sent_embeds = self.pooling(x, input_token_ids)

        return encoder_outputs, sent_embeds

    def forward_token(self, input_token_ids, attention_mask, output_layer=-1):
        if self.training:
            drop_mask = torch.bernoulli(
                self.config.word_dropout *
                torch.ones(input_token_ids.shape)).bool().cuda()
            input_token_ids = input_token_ids.masked_fill(drop_mask, 50264)

        input_token_embeds = self.embed_tokens(
            input_token_ids) + self.embed_positions(input_token_ids)
        x = self.layernorm_embedding(input_token_embeds)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)
        outputs = []
        for encoder_layer in self.layers:
            x, _ = encoder_layer(x, encoder_padding_mask=attention_mask)
            outputs.append(x)

        x = outputs[output_layer]
        x = x.transpose(0, 1)
        return x

    def forward_synt(self, input_synt_ids, attention_mask):
        input_synt_embeds = self.embed_synt(
            input_synt_ids) + self.embed_positions(input_synt_ids)
        y = self.synt_layernorm_embedding(input_synt_embeds)
        y = F.dropout(y, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        y = y.transpose(0, 1)

        for encoder_synt_layer in self.synt_layers:
            y, _ = encoder_synt_layer(y, encoder_padding_mask=attention_mask)

        # T x B x C -> B x T x C
        y = y.transpose(0, 1)
        return y

    def embed(self, input_token_ids, attention_mask=None, pool='mean'):
        if attention_mask is None:
            attention_mask = input_token_ids == self.config.pad_token_id

        x = self.forward_token(input_token_ids, attention_mask)
        # pdb.set_trace()
        sent_embeds = self.pooling(x, input_token_ids)
        return sent_embeds

    @classmethod
    def from_pretrained(cls, sem_extractor_model, bart_config_cache_dir):
        config = BartConfig.from_pretrained('facebook/bart-base',
                                            cache_dir=bart_config_cache_dir)
        state = torch.load(sem_extractor_model, map_location='cpu')
        shared = nn.Embedding(config.vocab_size, config.d_model,
                              config.pad_token_id)
        sem_extractor_encoder = cls(config, shared)
        sem_extractor_encoder.load_state_dict(state, strict=False)
        return sem_extractor_encoder


class MeanPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, input_token_ids):
        mask = input_token_ids != self.config.pad_token_id
        mean_mask = mask.float() / mask.float().sum(1, keepdim=True)
        x = (x * mean_mask.unsqueeze(2)).sum(1, keepdim=True)
        return x