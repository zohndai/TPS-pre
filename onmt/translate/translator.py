#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import time
import numpy as np
from itertools import count, zip_longest
import pandas as pd
import torch

import onmt.model_builder
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.translate.greedy_search import GreedySearch
from onmt.utils.misc import tile, set_random_seed, report_matrix
from onmt.utils.alignment import extract_alignment, build_align_pharaoh
from onmt.modules.copy_generator import collapse_copy_scores


def build_translator(opt, report_score=True, logger=None, out_file=None):

    log_probs_out_file=None
    target_score_out_file=None
    #print('out_file: {}'.format(out_file))
    if out_file is None:#True
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    if opt.log_probs:#False
        log_probs_out_file = codecs.open(opt.output + '_log_probs', 'w+', 'utf-8')
        target_score_out_file = codecs.open(opt.output + '_gold_score', 'w+', 'utf-8')

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_align=opt.report_align,
        report_score=report_score,
        logger=logger,
        log_probs_out_file=log_probs_out_file,
        target_score_out_file=target_score_out_file,
    )
    return translator


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=30,
            random_sampling_topk=1,
            random_sampling_temp=1,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            phrase_table="",
            data_type="text",
            verbose=False,
            report_time=False,
            copy_attn=False,
            global_scorer=None,
            out_file=None,
            log_probs_out_file=None,
            target_score_out_file=None,
            report_align=False,
            report_score=True,
            logger=None,
            seed=-1):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]#3
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]#1
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]#2
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]#0
        self._tgt_vocab_len = len(self._tgt_vocab)#74

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk

        self.min_length = min_length
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.log_probs_out_file = log_probs_out_file
        self.target_score_out_file = target_score_out_file
        self.report_align = report_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opt(
            cls,
            model,
            fields,
            opt,
            model_opt,
            global_scorer=None,# onmt.translate.GNMTGlobalScorer.from_opt(opt)
            out_file=None,
            log_probs_out_file=None,
            target_score_out_file=None,
            report_align=False,
            report_score=True,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_align (bool) : See :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        obj = cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,#-1
            n_best=opt.n_best,#1
            min_length=opt.min_length,#0
            max_length=opt.max_length,#3000
            ratio=opt.ratio,#-0.0
            beam_size=opt.beam_size,#5
            random_sampling_topk=opt.random_sampling_topk,#
            random_sampling_temp=opt.random_sampling_temp,#1
            stepwise_penalty=opt.stepwise_penalty,#False
            dump_beam=opt.dump_beam,#''
            block_ngram_repeat=opt.block_ngram_repeat,#0
            ignore_when_blocking=set(opt.ignore_when_blocking),#set([])
            replace_unk=opt.replace_unk,#False
            phrase_table=opt.phrase_table,#''
            data_type=opt.data_type,#'text
            verbose=opt.verbose,#False
            report_time=opt.report_time,#False
            copy_attn=model_opt.copy_attn,#False
            global_scorer=global_scorer,
            out_file=out_file,
            report_align=report_align,
            report_score=report_score,
            logger=logger,
            seed=opt.seed)#829
        obj.opt = opt
        return obj

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(self, batch, memory_bank, src_lengths, src_vocabs,
                    use_src_map, enc_states, batch_size, src):
       
        
        if "tgt" in batch.__dict__: #False
            
            gs = self._score_target(
                batch, memory_bank, src_lengths, src_vocabs,
                batch.src_map if use_src_map else None)
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:

            gs = [0] * batch_size
        #print("check gs value ....{}".format(gs))
        return gs

    def translate(
            self,
            src,
            tgt=None,
            src_dir=None,#''
            batch_size=None,#64
            batch_type="sents",#'sents'
            attn_debug=False,#false
            align_debug=False,#False
            phrase_table=""):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        #print("tgt_data:{}".format(tgt))
        #tgt = None
        if batch_size is None:#false
            raise ValueError("batch_size must be set")
        #src: [b'[O-] N = O . [OH]\r\n', b'[O] C C C C . [O] C C C C . O = [O+] [O-] . ...]
        src_data = {"reader": self.src_reader, "data": src, "dir": src_dir}
        tgt_data = {"reader": self.tgt_reader, "data": tgt, "dir": None}
        _readers, _data, _dir = inputters.Dataset.config(
            [('src', src_data), ('tgt', tgt_data)])
        #print("_reader:{}".format(_readers))
        #print("_data:1_len {}".format(len(_data)))
        #print("_data:##->{} ##-".format(_data[0][0]))
        #print("_dir:{}".format(_dir))
        #_readers:[self.src_reader, self.tgt_reader]
        #_data: [('src', src)]
        #_dir:['']

        #print('filter_pred:{}'.format(self._filter_pred))
        #print("ready to entering data......")
        data = inputters.Dataset(
            self.fields, readers=_readers, data=_data, dirs=_dir,
            sort_key=inputters.str2sortkey[self.data_type],#text_sort_key
            filter_pred=self._filter_pred #None
        )
        #print("ready to entering data_iteror......")
        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,#None
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        xlation_builder = onmt.translate.TranslationBuilder(
            data, self.fields, self.n_best, self.replace_unk, tgt,
            self.phrase_table
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for batch in data_iter:
            #batch:  [torchtext.data.batch.Batch of size 64]
                    #[.src]:('[torch.LongTensor of size 87x64x1]', '[torch.LongTensor of size 64]')
                    #[.indices]:[torch.LongTensor of size 64]
            #print("batch:{}".format(batch))
            # print("data.src_vocabs:{}".format(data.src_vocabs))
            # data.src_vocabs = []
            # attn_debug = false
            batch_data = self.translate_batch(
                batch, data.src_vocabs, attn_debug
            )

            #print("batch_batch: {}".format(batch_data))

            translations, confidence = xlation_builder.from_batch(batch_data)
            conf_dt = pd.DataFrame(confidence)
            conf_savepath = self.opt.output.replace(".txt", "_confidence.csv")
            conf_dt.to_csv(conf_savepath, header=False, index=False)
            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]

                #print("all scores:{}".format(all_scores))

                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                    if self.target_score_out_file is not None:
                        self.target_score_out_file.write(
                             str(trans.gold_score.item()) + '\n')
                        self.target_score_out_file.flush()

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                if self.report_align:
                    align_pharaohs = [build_align_pharaoh(align) for align
                                      in trans.word_aligns[:self.n_best]]
                    n_best_preds_align = [" ".join(align) for align
                                          in align_pharaohs]
                    n_best_preds = [pred + " ||| " + align
                                    for pred, align in zip(
                                        n_best_preds, n_best_preds_align)]
                all_predictions += [n_best_preds]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.log_probs_out_file is not None:
                     self.log_probs_out_file.write(
                         '\n'.join([str(t.item()) for t in trans.pred_scores[:self.n_best]]) + '\n')
                     self.log_probs_out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    if self.data_type == 'text':
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    output = report_matrix(srcs, preds, attns)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                if align_debug:
                    if trans.gold_sent is not None:
                        tgts = trans.gold_sent
                    else:
                        tgts = trans.pred_sents[0]
                    align = trans.word_aligns[0].tolist()
                    if self.data_type == 'text':
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(align[0]))]
                    output = report_matrix(srcs, tgts, align)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

        end_time = time.time()

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            self._log(msg)
            if tgt is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log("Average translation time (s): %f" % (
                total_time / len(all_predictions)))
            self._log("Tokens per second: %f" % (
                pred_words_total / total_time))

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores, all_predictions

    def _align_pad_prediction(self, predictions, bos, pad):
        """
        Padding predictions in batch and add BOS.

        Args:
            predictions (List[List[Tensor]]): `(batch, n_best,)`, for each src
                sequence contain n_best tgt predictions all of which ended with
                eos id.
            bos (int): bos index to be used.
            pad (int): pad index to be used.

        Return:
            batched_nbest_predict (torch.LongTensor): `(batch, n_best, tgt_l)`
        """
        dtype, device = predictions[0][0].dtype, predictions[0][0].device
        flatten_tgt = [best.tolist() for bests in predictions
                       for best in bests]
        paded_tgt = torch.tensor(
            list(zip_longest(*flatten_tgt, fillvalue=pad)),
            dtype=dtype, device=device).T
        bos_tensor = torch.full([paded_tgt.size(0), 1], bos,
                                dtype=dtype, device=device)
        full_tgt = torch.cat((bos_tensor, paded_tgt), dim=-1)
        batched_nbest_predict = full_tgt.view(
            len(predictions), -1, full_tgt.size(-1))  # (batch, n_best, tgt_l)
        return batched_nbest_predict

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        # (0) add BOS and padding to tgt prediction
        if hasattr(batch, 'tgt'):
            batch_tgt_idxs = batch.tgt.transpose(1, 2).transpose(0, 2)
        else:
            batch_tgt_idxs = self._align_pad_prediction(
                predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx)
        tgt_mask = (batch_tgt_idxs.eq(self._tgt_pad_idx) |
                    batch_tgt_idxs.eq(self._tgt_eos_idx) |
                    batch_tgt_idxs.eq(self._tgt_bos_idx))

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)

        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(src_len, batch * n_best, nfeat)``
        src = tile(src, n_best, dim=1)
        enc_states = tile(enc_states, n_best, dim=1)
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, n_best, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, n_best, dim=1)
        src_lengths = tile(src_lengths, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # reshape tgt to ``(len, batch * n_best, nfeat)``
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1)).T.unsqueeze(-1)
        dec_in = tgt[:-1]  # exclude last target from inputs
        _, attns = self.model.decoder(
            dec_in, memory_bank, memory_lengths=src_lengths, with_align=True)

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignement = extract_alignment(
            alignment_attn, prediction_mask, src_lengths, n_best)
        return alignement

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            #print("beam_size: {}".format(self.beam_size))
            if self.beam_size == 1:#False
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    batch_size=batch.batch_size,
                    min_length=self.min_length, max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk)
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                #print("exclusions_idx:{}".format(self._exclusion_idxs))
                decode_strategy = BeamSearch(
                    self.beam_size,#5
                    batch_size=batch.batch_size,#64
                    pad=self._tgt_pad_idx,#1
                    bos=self._tgt_bos_idx,#2
                    eos=self._tgt_eos_idx,#3
                    n_best=self.n_best,#5
                    global_scorer=self.global_scorer,#GNMTGlobalScorer.from_opt(opt)
                    min_length=self.min_length,#0
                     max_length=self.max_length,#3000
                    return_attention=attn_debug or self.replace_unk,#false
                    block_ngram_repeat=self.block_ngram_repeat,#0
                    exclusion_tokens=self._exclusion_idxs,#set()
                    stepwise_penalty=self.stepwise_penalty,#False
                    ratio=self.ratio)#-0.0
            return self._translate_batch_with_strategy(batch, src_vocabs,
                                                       decode_strategy)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)

        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            batch,
            src_vocabs,#[]
            memory_lengths,
            src_map=None,#None
            step=None,
            batch_offset=None):
        #print("src_vocabs,{} src_map,{} copy_attn{}".format(src_vocabs, src_map, self.copy_attn))
        if self.copy_attn:#False
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )
        #
        #tgt_words: tensor([[2,2...]], size([320,1]))
        #emb: tensor([[[]]], size([1, 320, 384])
        #output: tensor([[[]]],size([320, 1, 384]))
        #src_memory_bank:tensor([[[]]], size([batch_size*beam_size, seq_len, d_m]))
        #self.state:{"src": src, "cache": {"layer_0": {'memory_keys': None,
        #'memory_values': None, 'self_keys': None, 'self_values': None}, "layer_1": {}}}
        #src_lens: tensor([], size:([batch_size * beam_size]))
        #src_max_len=87
        #src_pad_mask


        #++++++++++++++++++++++++++++++++++
        # dec_out: tensor: size:[1, 320, 384]
        # dec_attn: dict: {'std': tensor:size:[1, 320, 87]}
        #
        #print("dec_out_shape:{}".format(dec_out.shape))
        #print("dec_attn_shape:{}".format(dec_attn['std'].shape))

        #
        #
        # Generator forward.

        if not self.copy_attn:# True: self.copy_attn==False
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
            log_probs = self.model.generator(dec_out.squeeze(0)/1.8)
            #print("log_probs:{}, size:{}".format(log_probs, log_probs.size()))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [tgt_len, batch_size, vocab] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(-1, batch.batch_size, scores.size(-1))
                scores = scores.transpose(0, 1).contiguous()
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        #print("log_probs1:{}, size:{}".format(log_probs, log_probs.size()))
        #print("log_probs:{}".format(log_probs.shape))

        # log_probs:tensor:size:[320, 74]

        return log_probs, attn

    def _translate_batch_with_strategy(
            self,
            batch,
            src_vocabs,
            decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn#false
        parallel_paths = decode_strategy.parallel_paths  # beam_size = 5
        batch_size = batch.batch_size#64
        #print("parallel_paths:{}".format(parallel_paths))
        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        #print("memory_bank2:{}, memory_bank_shape:{}".format(memory_bank, memory_bank.size()))
        self.model.decoder.init_state(src, memory_bank, enc_states)
        
        #print("self.model.decoder_state :{}".format(self.model.decoder.state['src'].size()))
        #print("batch_size : {}".format(batch_size))
        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}
            #gold : [0, 0, ...(一共64个0)]

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = batch.src_map if use_src_map else None
        fn_map_state, memory_bank, memory_lengths, src_map = \
            decode_strategy.initialize(memory_bank, src_lengths, src_map)
        #memory_bank：tensor([[[seq_len, batch_size * beam_size, d_m]]])
        # memory_lengths: tensor([], size:([batch_size * beam_size]))
        #src_map : None

        if fn_map_state is not None:#True
            self.model.decoder.map_state(fn_map_state)

        #src:tensor([seq_len, batch_size, 1])
        #self.decoder.state['src'] = tensor([seq_len, batch_size * beam_size, 1])

        #print("self.model.decoder.state['src']:{}, size :{}".format(self.model.decoder.state['src'], self.model.decoder.state['src'].size()))
        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            #print("decoder_input:{}, size:{}".format(decoder_input, decoder_input.size()))
            #tensor([[[2], [2]...[2]]], size:[1, 320, 1]) 全为2

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset)


            #tgt_words: tensor([[2,2...]], size([320,1]))
            #emb: tensor([[[]]], size([1, 320, 384])
            #output: tensor([[[]]],size([320, 1, 384]))
            #src_memory_bank:tensor([[[]]], size([batch_size*beam_size, seq_len, d_m]))
            #log_probs: tensor:size:[(batch_size x beam_size) , vocab]
            #attn = tensor:size:[1, 320, 87]
            #print("log_prob2:{}, size:{}, step:{}".format(log_probs, log_probs.shape, step))
            #print("attn:{}".format(attn))

            #print("attn:{}".format(attn.shape))
            #print("this is advance....")
            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        if self.report_align:
            results["alignment"] = self._align_forward(
                batch, decode_strategy.predictions)
        else:
            results["alignment"] = [[] for _ in range(batch_size)]
        #print("results: {}".format(results))
        return results

    def _score_target(self, batch, memory_bank, src_lengths,
                      src_vocabs, src_map):
        tgt = batch.tgt
        tgt_in = tgt[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in, memory_bank, batch, src_vocabs,
            memory_lengths=src_lengths, src_map=src_map)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            avg_score = score_total / words_total
            ppl = np.exp(-score_total.item() / words_total)
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, avg_score,
                name, ppl))
        return msg
