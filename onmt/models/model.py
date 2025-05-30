""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        #print("src:{}".format(src))
        #print("src.size:{}".format(src.size()))
        #print("tgt1:{}".format(tgt))
        #print("tgt1.size:{}".format(tgt.size()))

        dec_in = tgt[:-1]  # exclude last target from inputs
        #print("dec_in: {}".format(dec_in))
        #print("dec_in size:{}".format(dec_in.size()))

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        #print(f"memory_bank1: {memory_bank}, shape:{memory_bank.size()}")

        if bptt is False:#bptt=False
            #print("bptt is {}".format(bptt))#False
            #print("with_align is {}".format(with_align))#False

            #self.state["src"] = src
            #self.state["cache"] = None
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        #print("decoder_attns:=============={}+++++++++++".format(attns))
        #print("decoder_out:{}, shape:{}".format(dec_out, dec_out.shape))
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
