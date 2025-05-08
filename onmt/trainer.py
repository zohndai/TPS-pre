"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""
#from visdom import Visdom


import torch
import traceback

import onmt.utils
from onmt.utils.logging import logger


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           with_align=True if opt.lambda_align > 0 else False,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0]):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size #0
        self.shard_size = shard_size #32
        self.norm_method = norm_method #'tokens'
        self.accum_count_l = accum_count #[4]
        self.accum_count = accum_count[0] #4
        self.accum_steps = accum_steps #[0]
        self.n_gpu = n_gpu #1
        self.gpu_rank = gpu_rank #0
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align #False
        self.model_saver = model_saver #
        self.average_decay = average_decay #0
        self.moving_average = None
        self.average_every = average_every #1
        self.model_dtype = model_dtype # 'fp32'
        self.earlystopper = earlystopper #onmt.utils.EarlyStopping
        self.dropout = dropout #[0.1]
        self.dropout_steps = dropout_steps #[0]
        self.val_lo = 0
        self.val_ac = 0

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):

        
        
        for i in range(len(self.dropout_steps)):
            #print("==========================entering2")
            if step > 1 and step == self.dropout_steps[i] + 1:
                print("==========================entering3")
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)#4
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                #print("=====================================")
                #print("========{}".format(batch.tgt))
                #print("========{}".format(batch.tgt.size()))
                #print("____________{}".format(self.train_loss.padding_idx))
                #print("*********{},{}".format(batch.tgt[1, :, 0], batch.tgt[1, :, 0].size()))
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()#统计True的个数->tensor([])
                #print("num_tokens--------------{}{}".format(num_tokens, type(num_tokens)))
                #print("+++++++num_tolkenns{}".format(num_tokens.item()))

                normalization += num_tokens.item()

            else:
                normalization += batch.batch_size
            #print("######baches{}".format(batches))
            #print("######baches_len{}, self.accum_count{}".format(len(batches), self.accum_count))
            #print("-----------------normalization{}".format(normalization))
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if batches:
            print("enter-------------")
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """

        viz = Visdom()
        viz.line([[0., 0.]], [0.], win="loss", opts=dict(title="loss", legend=["training loss", "validate loss"]))
        viz.line([[0., 0.]], [0.], win="acc", opts=dict(title="acc", legend=["trianing acc", "validate acc"]))


        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            #print("===================self.optim.training_step{}".format(self.optim.training_step))
            step = self.optim.training_step#init=1
            #print("===================step{}".format(step))
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)#实际上无更新

            if self.gpu_verbose_level > 1:#False
                logger.info("GpuRank %d: index: %d" % (self.gpu_rank, i))
            if self.gpu_verbose_level > 0:#False
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:#Fasle
                normalization = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization))

            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:#False
                self._update_average(step)


            #print(f"report loss -- > {total_stats.loss}, step: {step}")

            #logger.info(f"report loss -- > {total_stats.loss}, step: {step}")

            train_loss = report_stats.loss/100000
            train_acc = report_stats.accuracy()

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            #logger.info("report loss -- > %.2f, step: %"%(report_stats.loss, step))
            #logger.info("report acc -- > %.2f stpe: %"%(report_stats.accuracy(), step))

            #logger.info("total loss -- > %.2f, step: %"%(total_stats.loss, step))
            #logger.info("total acc -- > %.2f stpe: %"%(total_stats.accuracy(), step))

            #print("step_report_*{}".format(report_stats.accuracy()))


            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:#False
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))

                #print('self.moving_average{}'.format(self.moving_average))

                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)

                
                if self.gpu_verbose_level > 0:#false
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))

                #print("gathering:{}".format(self.))

                valid_stats = self._maybe_gather_stats(valid_stats)

                if self.gpu_verbose_level > 0:#$false
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)


                self.val_lo = valid_stats.loss/100000
                self.val_ac = valid_stats.accuracy()

                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break
            #if step % 50 == 0:
                #viz.line([[train_loss, train_acc]], [step], win="pre-train", update="append")
            if step % 10 == 0:
                viz.line([[train_loss, self.val_lo]], [step], win="loss", update="append")
                viz.line([[train_acc, self.val_ac]], [step], win="acc", update="append")


            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                      and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)
                # self.moving_average=None
            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)


        return total_stats




    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.

        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths,
                                             with_align=self.with_align)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)

                

        if moving_average:
            for param_data, param in zip(model_params_data,
                                         self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        #logger.info("total loss -- > %.2f, step: %" % (stats.loss, step))
        #logger.info("total acc -- > %.2f stpe: %" % (stats.accuracy(), step))
        #logger.info('validation loss --> %.2f' % (stats.loss))


        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:#True

            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):

            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            #print("self.trunc_size$$$$${}".format(self.trunc_size))
            if self.trunc_size:#False
                trunc_size = self.trunc_size
                #print("%%%%%%%%%%%%%trunc_size{}".format(trunc_size))
            else:
                trunc_size = target_size
                #print("%%%%%%%%%%%%%trunc_size2{}".format(trunc_size))
            #print("=={}==={}".format(batch.src, type(batch.src)))

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)#True
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt
            #print("=={}==={}".format(batch.tgt, batch.tgt.size()))

            bptt = False
            for j in range(0, target_size-1, trunc_size):
                #print("j=={}===".format(j)) j===0

                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]#tgt=tgt_out

                # 2. F-prop all but generator.
                #print("&&&&&&{}&&&&&&".format(self.accum_count))
                if self.accum_count == 1:#False
                    #print("self.accum_count++++{}++++".format(self.accum_count))
                    self.optim.zero_grad()

                outputs, attns = self.model(src, tgt, src_lengths, bptt=bptt,
                                            with_align=self.with_align)
                bptt = True

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,#32
                        trunc_start=j,
                        trunc_size=trunc_size)
                    #print("training loss______{}".format(loss))

                    if loss is not None: #loss = None
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)

                # 4. Update the parameters and statistics.

                if self.accum_count == 1:#False
                    
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    
                    self.optim.step()
                    

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                #print("********{}".format(self.model.decoder.state))
                if self.model.decoder.state is not None:#self.model.decoder.state = {"src":tensors, cache:None}
                    self.model.decoder.detach_state()#self.model.decoder.state["src"].detach()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            #print("###############")
            self.optim.step()
            #print("@@@@@@@@@@@@@@@{}".format(self.optim._fp16))

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        #print("entering:maybe_gather_stats_1")
        if stat is not None and self.n_gpu > 1:#False
            print("entering:maybe_gather_stats_2")
            return onmt.utils.Statistics.all_gather_stats(stat)
        #print("entering:maybe_gather_stats_3")
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
    #step, train_steps,
    #self.optim.learning_rate(),report_stats)
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        #print(f"self.report_manager: {self.report_manager}")
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
