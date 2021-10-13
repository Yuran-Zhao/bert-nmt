import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bart import (PretrainedBartModel, LayerNorm,
                                        EncoderLayer, DecoderLayer,
                                        LearnedPositionalEmbedding,
                                        _prepare_bart_decoder_inputs,
                                        _make_linear_from_emb)
from transformers import BartConfig, BartModel
from modules import SelfAttention, ACT2FN
import pdb

from parabart import ParaBartEncoder, MeanPooling, Discriminator


class SemExtractor(PretrainedBartModel):
    def __init__(self, config):
        super().__init__(config)

        self.cosine = nn.CosineSimilarity()

        self.shared = nn.Embedding(config.vocab_size, config.d_model,
                                   config.pad_token_id)

        self.encoder = ParaBartEncoder(config, self.shared)
        self.decoder = MixedCrossAttentionDecoder(config, self.shared)

        self.linear = nn.Linear(config.d_model, config.vocab_size)

        self.adversary = Discriminator(config)

        self.init_weights()

    def forward(self,
                input_ids,
                decoder_input_ids,
                attention_mask=None,
                decoder_padding_mask=None,
                encoder_outputs=None,
                return_encoder_outputs=False):
        """[summary]

        Args:
            input_ids (torch.Tensor): ```(batch_size, max_sent_len + 2 + max_synt_len + 2)```
            decoder_input_ids (torch.Tensor): ```(batch_size, max_sent_len + 2)```
            attention_mask (torch.Tensor, optional): mask for encoder, shape is the same as input_ids. Defaults to None.
            decoder_padding_mask (Torch.Tensor, optional): [description]. Defaults to None.
            encoder_outputs (Tuple(Torch.Tensor, Torch.Tensor), optional): 
                sem_encoder_outputs || syn_encoder_outputs: ```(batch_size, max_sent_len + 2 + max_synt_len + 2, d_model)```
                mean_pooled_sem_outputs: ```(batch_size, d_model)```
                Defaults to None.
            return_encoder_outputs (bool, optional): 
                only invoked to get the result of encoder or not. Defaults to False.

        Returns:
            [type]: [description]
        """
        if attention_mask is None:
            attention_mask = input_ids == self.config.pad_token_id

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids,
                                           attention_mask=attention_mask)

        if return_encoder_outputs:
            return encoder_outputs

        assert encoder_outputs is not None
        assert decoder_input_ids is not None

        decoder_input_ids = decoder_input_ids[:, :-1]

        _, decoder_padding_mask, decoder_causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids=None,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask,
            causal_mask_dtype=self.shared.weight.dtype,
        )

        # if we make use of sem_outputs and syn_outputs at the same time
        # the decoder attentino is the same as `attention_mask`
        # attention_mask2 = torch.cat(
        #     (torch.zeros(input_ids.shape[0], 1).bool().cuda(),
        #      attention_mask[:, self.config.max_sent_len + 2:]),
        #     dim=1)

        # decoder
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=decoder_causal_mask,
            encoder_attention_mask=attention_mask,
        )

        batch_size = decoder_outputs.shape[0]
        outputs = self.linear(decoder_outputs.contiguous().view(
            -1, self.config.d_model))
        outputs = outputs.view(batch_size, -1, self.config.vocab_size)

        # discriminator
        for p in self.adversary.parameters():
            p.required_grad = False
        adv_outputs = self.adversary(encoder_outputs[1])

        return outputs, adv_outputs

    # def prepare_inputs_for_generation(self, decoder_input_ids, past,
    #                                   attention_mask, use_cache, **kwargs):
    #     assert past is not None, "past has to be defined for encoder_outputs"

    #     encoder_outputs = past[0]
    #     return {
    #         "input_ids":
    #         None,  # encoder_outputs is defined. input_ids not needed
    #         "encoder_outputs":
    #         encoder_outputs,
    #         "decoder_input_ids":
    #         torch.cat(
    #             (decoder_input_ids,
    #              torch.zeros(
    #                  (decoder_input_ids.shape[0], 1), dtype=torch.long).cuda()),
    #             1),
    #         "attention_mask":
    #         attention_mask,
    #     }

    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)

    def get_input_embeddings(self):
        return self.shared

    @staticmethod
    def _reorder_cache(past, beam_idx):
        enc_out = past[0][0]

        new_enc_out = enc_out.index_select(0, beam_idx)

        past = ((new_enc_out, ), )
        return past

    def forward_adv(self,
                    input_token_ids,
                    attention_mask=None,
                    decoder_padding_mask=None):
        for p in self.adversary.parameters():
            p.required_grad = True
        sent_embeds = self.encoder.embed(
            input_token_ids, attention_mask=attention_mask).detach()
        adv_outputs = self.adversary(sent_embeds)

        return adv_outputs

    def compute_similarity(self, sent1_token_ids, sent2_token_ids):
        sent1_embeds = self.encoder.embed(sent1_token_ids).detach()
        sent2_embeds = self.encoder.embed(sent2_token_ids).detach()
        sent1_embeds = torch.squeeze(sent1_embeds, 1)
        sent2_embeds = torch.squeeze(sent2_embeds, 1)
        similarity = self.cosine(sent1_embeds, sent2_embeds)
        return nn.functional.softmax(similarity, dim=0)

    def init_weights(self):
        pass


class MixedCrossAttentionDecoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()

        self.dropout = config.dropout

        self.embed_tokens = embed_tokens

        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id,
            config.extra_pos_embeddings)

        self.layers = nn.ModuleList(
            [MixedCrossAttentionDecoderLayer(config) for _ in range(1)])
        self.layernorm_embedding = LayerNorm(config.d_model)

    def forward(self, decoder_input_ids, encoder_hidden_states,
                decoder_padding_mask, decoder_causal_mask,
                encoder_attention_mask):
        """[summary]

        Args:
            decoder_input_ids (torch.Tensor): ]
                [batch_size, max_sent_len + 2]
            encoder_hidden_states (toch.Tensor): 
                sem_encoder_outputs || syn_encoder_outputs
                [batch_size, max_sent_len + 2 + max_synt_len + 2, d_model]
            decoder_padding_mask (torch.Tensor): 
                [batch_size, max_sent_len + 2]
            decoder_causal_mask ([type]):
                [batch_size, max_sent_len + 2]
            encoder_attention_mask ([type]):
                [batch_size, max_sent_len + 2 + max_synt_len + 2]

        Returns:
            [type]: [description]
        """

        x = self.embed_tokens(decoder_input_ids) + self.embed_positions(
            decoder_input_ids)
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # make the batch_size to the second dimention
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        for idx, decoder_layer in enumerate(self.layers):
            x, _, _ = decoder_layer(x,
                                    encoder_hidden_states,
                                    encoder_attn_mask=encoder_attention_mask,
                                    decoder_padding_mask=decoder_padding_mask,
                                    causal_mask=decoder_causal_mask)

        x = x.transpose(0, 1)

        return x


class MixedCrossAttentionDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.beta = config.beta
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            sem_encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.syn_encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            syn_encoder_decoder_attention=True,
        )
        self.syn_encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
    ):
        """[summary]

        Args:
            x (torch.Tensor): 
                [batch_size, max_sent_len + 2, d_model]
            encoder_hidden_states (torch.Tensor):
                [batch_size, max_sent_len + 2 + max_synt_len + 2, d_model]
            encoder_attn_mask (torch.Tensor, optional): 
                [batch_size, max_sent_len + 2 + max_synt_len + 2]
                Defaults to None.
            layer_state (dict, optional): 
                [description]. 
                Defaults to None.
            causal_mask (torch.Tensor, optional): 
                [batch_size, max_sent_len + 2]. 
                Defaults to None.
            decoder_padding_mask (torch.Tensor, optional): 
                [batch_size, max_sent_len + 2]. 
                Defaults to None.
            output_attentions (bool, optional): 
                whether output the attention score or not. 
                Defaults to False.

        Returns:
            [type]: [description]
        """
        sem_encoder_hidden_state, syn_encoder_hidden_state = torch.split(
            encoder_hidden_states,
            [self.config.max_sent_len + 2, self.config.max_synt_len + 2],
            dim=0)

        sem_encoder_attn_mask, syn_encoder_attn_mask = torch.split(
            encoder_attn_mask,
            [self.config.max_sent_len + 2, self.config.max_synt_len + 2],
            dim=1)

        residual = x

        if layer_state is None:
            layer_state = {}

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Self Attention
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        # Semantic Cross Attention
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        # pdb.set_trace()
        sem_x, _ = self.encoder_attn(
            query=x,
            key=sem_encoder_hidden_state,
            key_padding_mask=sem_encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        sem_x = F.dropout(sem_x, p=self.dropout, training=self.training)
        sem_x = residual + sem_x
        if not self.normalize_before:
            sem_x = self.encoder_attn_layer_norm(sem_x)

        x = residual
        # Syntactic Cross Attention
        assert self.syn_encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.syn_encoder_attn_layer_norm(x)
        syn_x, _ = self.syn_encoder_attn(
            query=x,
            key=syn_encoder_hidden_state,
            key_padding_mask=syn_encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        syn_x = F.dropout(syn_x, p=self.dropout, training=self.training)
        syn_x = residual + syn_x
        if not self.normalize_before:
            syn_x = self.syn_encoder_attn_layer_norm(syn_x)

        # Mixed Cross Attention
        x = self.beta * sem_x + (1 - self.beta) * syn_x

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


if __name__ == '__main__':
    bart = BartModel.from_pretrained('facebook/bart-base',
                                     cache_dir='./bart-base/')
    print("=" * 10 + "Params in BART" + '=' * 10)
    for key in bart.state_dict().keys():
        print(key)
    config = BartConfig.from_pretrained('facebook/bart-base',
                                        cache_dir='./bart-base/')
    config.word_dropout = 0
    config.max_sent_len = 40
    config.max_synt_len = 50
    config.syntax_encoder_layer_num = 1
    config.rank = 32
    config.use_GAT = False
    config.beta = 0.5
    model = SemExtractor(config)
    print("=" * 10 + "Params in Sem-Extractor" + '=' * 10)
    for key, _ in model.state_dict().items():
        print(key)
    # model.load_state_dict(bart.state_dict(), strict=False)
    # print("=" * 10 + 'After Loading' + '=' * 10)
    # for key, _ in model.state_dict().items():
    #     print(key)
