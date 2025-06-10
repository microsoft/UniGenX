# -*- coding: utf-8 -*-
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GreedySearchOutput, SampleOutput
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.utils import ModelOutput

from unigenx.logging import logger

from .diffloss import DiffLoss


@dataclass
class UniGenXOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_log: Optional[dict] = None
    logits: Optional[torch.FloatTensor] = None
    coordinates: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    num_examples: Optional[int] = None
    label: Optional[torch.LongTensor] = None
    log_output: Optional[dict] = None


@dataclass
class UniGenXOutput(ModelOutput):
    sequences: torch.LongTensor = None
    coordinates: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UniGenX(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.coordinate_encoder = MLP(3, config.hidden_size, config.hidden_size)
        print("config.is_solver:",config.is_solver)
        print("config.solver_order:",config.solver_order)
        print("config.solver_type:",config.solver_type)
        self.diffloss = DiffLoss(
            target_channels=3,
            z_channels=config.hidden_size,
            width=config.diff_width,
            depth=config.diff_depth,
            num_sampling_steps=config.diff_steps,
            grad_checkpointing=False,
            is_solver=config.is_solver,
            dpm_num_sampling_steps = config.solver_steps,
            dpm_order = config.solver_order,
            solver_type = config.solver_type,
            algorithm_type = config.algorithm_type
        )
        self.diffusion_batch_mul = config.diff_mul
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_coordinates: torch.FloatTensor = None,
        coordinates_mask: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ntokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, UniGenXOutputWithPast]:
        r"""
        Args:
            label_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]`, or -100 (see `input_ids` docstring), or coordinates (x, y, z). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]` and coordinates.

        Returns:

        Example:

        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            words_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = words_embeds
            if input_coordinates is not None:
                if input_coordinates.dtype != words_embeds.dtype:
                    input_coordinates = input_coordinates.to(words_embeds.dtype)
                coordinates_embeds = self.coordinate_encoder(input_coordinates)
                inputs_embeds[coordinates_mask.bool()] = coordinates_embeds
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # pass the hidden states to lm_head and coordinate_decoder to get logits and coordinates
        # get word logits using lm head
        word_logits = self.lm_head(hidden_states)
        # get coordinates using coordinate decoder
        coordinates = hidden_states
        # coordinates = self.coordinate_decoder(hidden_states)

        # diffussion process
        if self.training or coordinates.shape[1] != 1:
            # shift so that tokens < n predict n
            y = input_coordinates
            shift_coordinates_mask = coordinates_mask[:, 1:].contiguous()
            y_hat = coordinates[:, :-1, :].contiguous()
            y_hat = y_hat[
                shift_coordinates_mask.bool()
            ]  # y_hat is the condition of the diffusion model
            y_hat = y_hat.repeat(self.diffusion_batch_mul, 1)
            y = y.repeat(
                self.diffusion_batch_mul, 1
            )  # apply different time to the same coordinates
            diffloss = self.diffloss(target=y, z=y_hat)

        else:
            # noise = self.diffusion.get_noise()
            y_0 = None

        if not return_dict:
            output = (word_logits, coordinates, y_0) + outputs[1:]
            return output

        return UniGenXOutputWithPast(
            loss=diffloss,
            logits=word_logits,
            coordinates=coordinates,
            # y_0=y_0,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    @torch.no_grad()
    def sample(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_coordinates: Optional[torch.FloatTensor] = None,
        coordinates_mask: Optional[torch.LongTensor] = None,
        only_seq: Optional[bool] = False,
        output_coordinates_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            words_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = words_embeds
            if input_coordinates is not None:
                if input_coordinates.dtype != words_embeds.dtype:
                    input_coordinates = input_coordinates.to(words_embeds.dtype)
                coordinates = input_coordinates[coordinates_mask.bool()]
                coordinates_embeds = self.coordinate_encoder(coordinates)
                inputs_embeds[coordinates_mask.bool()] = coordinates_embeds

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        word_logits = self.lm_head(hidden_states)

        coordinates = hidden_states

        y_hat = coordinates[:, -1:, :].contiguous()

        bsz, seq_len, _ = y_hat.shape
        if not only_seq:
            y_t = self.diffloss.sample(y_hat.reshape(bsz * seq_len, -1)).reshape(
                bsz, seq_len, -1
            )
        else:
            y_t = None
        if not return_dict:
            output = (word_logits, coordinates, y_t) + outputs[1:]
            return output

        return UniGenXOutputWithPast(
            logits=word_logits,
            coordinates=y_t,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        mask_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        only_seq = model_kwargs.get("only_seq", False)
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.generation_config.eos_token_id
        )
        mask_token_id = (
            mask_token_id if mask_token_id is not None else self.config.mask_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device)
            if eos_token_id is not None
            else None
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init input coordinates
        input_coordinates = torch.empty(
            (input_ids.shape[0], input_ids.shape[1], 3),
            dtype=torch.float32,
            device=input_ids.device,
        )
        if model_kwargs.get("input_coordinates", None) is not None:
            input_coordinates_mask = model_kwargs["coordinates_mask"][
                :, : input_ids.shape[1]
            ]
            input_coordinates[input_coordinates_mask.bool()] = model_kwargs[
                "input_coordinates"
            ]
            del model_kwargs["input_coordinates"]
        original_coordinates_mask = model_kwargs["coordinates_mask"].clone()

        # init attention / hidden states / scores tuples
        scores = None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, input_coordinates, **model_kwargs
            )

            # forward pass to get next token
            outputs = self.sample(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            coordinates_mask = model_inputs["output_coordinates_mask"]
            next_token_logits = outputs.logits[:, -1, :]
            if not only_seq:
                next_coordinates = outputs.coordinates[:, -1, :]
            next_coordinates_mask = coordinates_mask[:, -1]
            # else:
            #     next_coordinates_mask = torch.zeros_like(coordinates_mask[:, -1], dtype=torch.bool, device=coordinates_mask.device)

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_word = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if not only_seq:
                next_word.fill_(eos_token_id[0])

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_word = next_word * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            next_tokens = (
                next_word * (1 - next_coordinates_mask.bool().long())
                + mask_token_id * next_coordinates_mask.bool().long()
            )

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if not only_seq:
                input_coordinates = torch.cat(
                    [input_coordinates, next_coordinates[:, None]], dim=1
                )

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            # as stopping criteria checks the last dimension as the length, we pass the first value of the last dimension of the 3-d tensor
            if all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if not only_seq:
            input_coordinates = input_coordinates[original_coordinates_mask.bool()]
        else:
            input_coordinates = None

        if return_dict_in_generate:
            return UniGenXOutput(
                sequences=input_ids,
                coordinates=input_coordinates,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return (input_ids, input_coordinates)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        logits_warper = (
            logits_warper if logits_warper is not None else LogitsProcessorList()
        )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.generation_config.eos_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device)
            if eos_token_id is not None
            else None
        )
        output_scores = (
            output_scores
            if output_scores is not None
            else self.generation_config.output_scores
        )
        output_logits = (
            output_logits
            if output_logits is not None
            else self.generation_config.output_logits
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init input coordinates
        input_coordinates = torch.empty(
            (input_ids.shape[0], input_ids.shape[1], 3),
            dtype=torch.float32,
            device=input_ids.device,
        )
        if model_kwargs.get("input_coordinates", None) is not None:
            input_coordinates_mask = model_kwargs["coordinates_mask"][
                :, : input_ids.shape[1]
            ]
            input_coordinates[input_coordinates_mask.bool()] = model_kwargs[
                "input_coordinates"
            ]
            del model_kwargs["input_coordinates"]
        model_kwargs["coordinates_mask"].clone()

        # init attention / hidden states / scores tuples
        scores = None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )

        this_peer_finished = False  # used by synced_gpus only

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, input_coordinates, **model_kwargs
            )

            # forward pass to get next token
            outputs = self.sample(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            model_inputs["output_coordinates_mask"]
            next_token_logits = outputs.logits[:, -1, :]
            # next_coordinates = outputs.coordinates[:, -1, :]
            # next_coordinates_mask = coordinates_mask[:, -1]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                # if output_logits:
                #     raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            # as stopping criteria checks the last dimension as the length, we pass the first value of the last dimension of the 3-d tensor
            if all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        # input_coordinates = input_coordinates[original_coordinates_mask.bool()]

        if return_dict_in_generate:
            return UniGenXOutput(
                sequences=input_ids,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return (input_ids, input_coordinates)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_coordinates=None,
        coordinates_mask=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        only_seq=None,
        **kwargs,
    ):
        seq_length = input_ids.shape[1]
        input_coordinates_mask = coordinates_mask[:, :seq_length]
        output_coordinates_mask = coordinates_mask[:, 1 : seq_length + 1]
        if past_key_values:
            input_ids = input_ids[:, -1:]
            input_coordinates_mask = input_coordinates_mask[:, -1:]
            output_coordinates_mask = output_coordinates_mask[:, -1:]
            input_coordinates = input_coordinates[
                :, -1:
            ]  # [input_coordinates_mask.bool()]
        else:
            input_coordinates = input_coordinates  # [input_coordinates_mask.bool()]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "input_coordinates": input_coordinates,
                "coordinates_mask": input_coordinates_mask,
                "output_coordinates_mask": output_coordinates_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "only_seq": only_seq,
            }
        )
        return model_inputs
