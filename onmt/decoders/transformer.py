"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, full_context_alignment=False,
                 alignment_heads=None):
        super(TransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":#True
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)

        elif self_attn_type == "average":#False

            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment#False
        self.alignment_heads = alignment_heads#None

    def forward(self, *args, **kwargs):
        """ Extend _forward for (possibly) multiple decoder pass:
        1. Always a default (future masked) decoder forward pass,
        2. Possibly a second future aware decoder pass for joint learn
            full context alignement.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, 1, model_dim)``
            * top_attn ``(batch_size, 1, src_len)``
            * attn_align ``(batch_size, 1, src_len)`` or None
        """
        with_align = kwargs.pop('with_align', False)#False
        #print("with_align:{}".format(with_align))
        output, attns = self._forward(*args, **kwargs)#attns:tensor[batch_size, h_n, tgt_len, src_len]
        #print("attns.size:{}".format(attns.size()))
        top_attn = attns[:, 0, :, :].contiguous()#tensor[batch_size, tgt_len, src_len]
        #print("top_attn size:{}".format(top_attn.size()))
        attn_align = None
        if with_align:#False
            #print("with_align is not True:{}---------".format(with_align))
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads is not None:
                attns = attns[:, :self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return output, top_attn, attn_align#=None

    def _forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                 layer_cache=None, step=None, future=False):
        """ A naive forward pass for transformer decoder.
        # TODO: change 1 to T as T could be 1 or tgt_len
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attns ``(batch_size, head, 1, src_len)``

        """
        #print("memory_bank :{}".format(memory_bank.size()))
        #print("inputs_size:{}".format(inputs.size()))
        #print("tgt_pad_mask:{}".format(tgt_pad_mask.size()))
        #print("src_pad_mask:{}".format(src_pad_mask.size()))
        dec_mask = None

        if step is None:# translate:False
            
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        input_norm = self.layer_norm_1(inputs)#tgt[-1]
        #pre:tensor([[[]]],size([320, 1, 384]))

        if isinstance(self.self_attn, MultiHeadedAttention):
            #print("tgt_query:self start, layer_cache:{}".format(layer_cache))


            query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                      mask=dec_mask,
                                      layer_cache=layer_cache,#None
                                      attn_type="self")

            #print("tgt_query:self end, layer_cache:{}".format(layer_cache))

        elif isinstance(self.self_attn, AverageAttention):#False
            query, _ = self.self_attn(input_norm, mask=dec_mask,
                                      layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        #print("tgt_memory attn:context start, layer_cache:{}".format(layer_cache))

        #print("memory_shape:{}".format(memory_bank.shape))

        mid, attns = self.context_attn(memory_bank, memory_bank, query_norm,
                                       mask=src_pad_mask,
                                       layer_cache=layer_cache,
                                       attn_type="context")
        #print("tgt_memory attn:context end, layer_cache:{}".format(layer_cache))
        output = self.feed_forward(self.drop(mid) + query)

        return output, attns

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads=None):
        super(TransformerDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,#'scal-dot
             max_relative_positions=max_relative_positions,#0
             aan_useffn=aan_useffn,#False
             full_context_alignment=full_context_alignment,#False
             alignment_heads=alignment_heads)#None
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,#4
            opt.dec_rnn_size,# 384
            opt.heads,#8
            opt.transformer_ff,#2084
            opt.copy_attn,#False
            opt.self_attn_type,#scale-dot
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,#[0.1]
            opt.attention_dropout[0] if type(opt.attention_dropout)#[0.1]
            is list else opt.dropout,
            embeddings, #tgt_emb
            opt.max_relative_positions,#0
            opt.aan_useffn,#False
            opt.full_context_alignment,#False
            opt.alignment_layer,#-3
            alignment_heads=opt.alignment_heads)#None

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):# (dec_in, memory_bank,
                                                            # memory_lengths=lengths,
                                                       # with_align=with_align#False)
        """Decode, possibly stepwise."""
        if step == 0:#False
            self._init_cache(memory_bank)
        # self.state:{"src": src, "cache": {"layer_0": {'memory_keys': None,
        # 'memory_values': None, 'self_keys': None, 'self_values': None}, "layer_1": {}}}

        # state->{"cache":{"layer_0": layer_cache}, "layer_1:{"memory_keys": None, "memory_values": None,
        # "self_keys":None}}
        #print("tgt: {}".format(tgt))
        #print("tgt.size(): {}".format(tgt.size()))

        #print("memory_bank:{}".format(memory_bank))
        #print(memory_bank.size())

        tgt_words = tgt[:, :, 0].transpose(0, 1)#tensor([batch_size × seq_len])
        #print("tgt words----------{}, tgt-size:{}".format(tgt_words, tgt_words.size()))


        emb = self.embeddings(tgt, step=step)
        #print("emb:{}, size :{}".format(emb, emb.size()))
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()



        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx#1

        src_lens = kwargs["memory_lengths"]
        #print("src_lens:{}, src_lens_size:{}".format(src_lens, src_lens.size()))
        src_max_len = self.state["src"].shape[0]

        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        # == ~sequence_mask(src_lens).unsqueeze(1)
        #print("src_pad_mask:{}, size:{}".format(src_pad_mask, tgt_pad_mask.size()))
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        #print("tgt_pas_mask:{}, size:{}".format(tgt_pad_mask, tgt_pad_mask.size()))

        #因为tgt中包含起始符号，因此不能使用和src相同的padd方案

        with_align = kwargs.pop('with_align', False)#False
        #print("step:{}".format(step))
        attn_aligns = []

        for i, layer in enumerate(self.transformer_layers):

            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None #None
            # layer_cache:{'memory_keys': None,
            # 'memory_values': None, 'self_keys': None, 'self_values': None}

            # print("tgt_shape:{}".format(output.shape))
            #print("memory_shape:{}".format(src_memory_bank.shape))
            #print("step:{}".format(step))
            #print("this is decoder****************")
            output, attn, attn_align = layer(
                output,# [batch_size × seq_len × d_m],pre:tensor([[[]]],size([320, 1, 384]))
                src_memory_bank, #batch_size×seq_len×d_m pre:tensor([[[]]],size([batch_size*beam_size,seq_len,d_m]))
                src_pad_mask, # ==src_mask [batch_size, 1, src_seq_len]
                tgt_pad_mask, #tgt_mask  [batch_size, 1, tgt_seq_len]
                layer_cache=layer_cache, # None
                step=step,#None
                with_align=with_align) # False
            #print("this is decoder+++++++++++++++++++")
            # print("dec_out_shape:{}".format(output.shape))
            # print("top_attn:{}".format(attn.shape))
            # print("attn_align:{}".format(attn_align))
            if attn_align is not None: # attn_align==None
                attn_aligns.append(attn_align)


        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        #dec_outs: tensor:size(1, 320, 384)
        #print("dec_out_shape:{}".format(dec_outs.shape))
        attn = attn.transpose(0, 1).contiguous()
        #tensor:size:[1, 320, 87]

        #print("attns_shape:{}".format(attn.shape))

        attns = {"std": attn}
        
        #print("check------------------{}".format(self._copy))
        if self._copy:#False
            #print(".copy is True+++++++")
            attns["copy"] = attn
        #print("check: {}".format(with_align))
        if with_align:#False
            #print("with_align is True*******")
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1) # batch_size, translate: batch_szie * beam_size
        depth = memory_bank.size(-1) # d_m

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            #print("is entering layer is average...")
            if isinstance(layer.self_attn, AverageAttention):#False
                #print("layer is average...")
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            #state->{"cache":{"layer_0": layer_cache},
            # {"layer_1" :{"memory_keys": None, "memory_values":None,
            # "self_keys":None, "self.values":None}}

            self.state["cache"]["layer_{}".format(i)] = layer_cache
            # self.state:{"src": src, "cache": {"layer_0": {'memory_keys': None,
            # 'memory_values': None, 'self_keys': None, 'self_values': None}, "layer_1": {}}}



    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
