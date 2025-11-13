#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from ucm.logger import init_logger

logger = init_logger(__name__)


# ==================== vllm/v1/request.py ====================
def _patch_request_succeed_dumped_blocks() -> None:
    """Patch Request to add succeed_dumped_blocks field."""
    try:
        from vllm.v1.request import Request

        original_init = Request.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.succeed_dumped_blocks = []

        Request.__init__ = __init__
    except ImportError:
        logger.warning("Could not patch Request.__init__ - module not found")


# ==================== vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py ====================
def _patch_multi_connector_wait_for_save() -> None:
    """Patch MultiConnector to add wait_for_save method."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )

        def wait_for_save(self):
            success_dumped_blocks = None
            for c in self._connectors:
                uc_dump_blocks = c.wait_for_save()
                if uc_dump_blocks:
                    success_dumped_blocks = uc_dump_blocks

            return success_dumped_blocks if success_dumped_blocks else None

        MultiConnector.wait_for_save = wait_for_save
    except ImportError:
        logger.warning(
            "Could not patch MultiConnector.wait_for_save - module not found"
        )


def _patch_outputs_finished_dumping() -> None:
    """Patch ModelRunnerOutput to add finished_dumping field."""
    try:
        from dataclasses import field, make_dataclass
        from typing import Optional

        from vllm.v1.outputs import ModelRunnerOutput

        patched_ModelRunnerOutput = make_dataclass(
            "ModelRunnerOutput",
            [
                (
                    "finished_dumping",
                    Optional[dict[str, list[str]]],
                    field(default=None),
                ),
                ("invalid_block_ids", set[int], field(default_factory=set)),
            ],
            bases=(ModelRunnerOutput,),
        )
        ModelRunnerOutput = patched_ModelRunnerOutput
    except ImportError:
        logger.warning("Could not patch ModelRunnerOutput - module not found")


def _patch_scheduler_update_from_output() -> None:
    """Patch Scheduler to add finished_dumping handling."""
    try:
        from collections import defaultdict
        from typing import Optional

        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )
        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.sched.utils import check_stop
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.request import Request
        from vllm.v1.spec_decode.metrics import SpecDecodingStats

        def update_from_output(
            self,
            scheduler_output: SchedulerOutput,
            model_runner_output: ModelRunnerOutput,
        ) -> dict[int, EngineCoreOutputs]:
            sampled_token_ids = model_runner_output.sampled_token_ids
            spec_token_ids = model_runner_output.spec_token_ids
            logprobs = model_runner_output.logprobs
            prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            pooler_outputs = model_runner_output.pooler_output
            num_nans_in_logits = model_runner_output.num_nans_in_logits

            new_running: list[Request] = []
            outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
            spec_decoding_stats: Optional[SpecDecodingStats] = None

            # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
            # loop can be a performance bottleneck. We should do our best to avoid
            # expensive operations inside the loop.
            for request in self.running:
                req_id = request.request_id
                num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
                if num_tokens_scheduled == 0:
                    # The request was not scheduled in this step.
                    new_running.append(request)
                    continue

                req_index = model_runner_output.req_id_to_index[req_id]
                generated_token_ids = (
                    sampled_token_ids[req_index] if sampled_token_ids else []
                )

                scheduled_spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
                )
                if scheduled_spec_token_ids:
                    # num_computed_tokens represents the number of tokens
                    # processed in the current step, considering scheduled
                    # tokens and rejections. If some tokens are rejected,
                    # num_computed_tokens is decreased by the number of rejected
                    # tokens, where is given by:
                    # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                    num_tokens_rejected = (
                        len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)
                    )
                    request.num_computed_tokens -= num_tokens_rejected
                    spec_decoding_stats = self.make_spec_decoding_stats(
                        spec_decoding_stats,
                        num_draft_tokens=len(scheduled_spec_token_ids),
                        num_accepted_tokens=len(generated_token_ids) - 1,
                    )

                # NOTE(woosuk): This has to be executed after updating
                # `request.num_computed_tokens`.
                if request.has_encoder_inputs:
                    self._free_encoder_inputs(request)

                stopped = False
                new_logprobs = None
                new_token_ids = generated_token_ids
                kv_transfer_params = None
                if model_runner_output.finished_dumping is not None:
                    request.succeed_dumped_blocks.extend(
                        model_runner_output.finished_dumping.get(req_id, [])
                    )
                    is_prefill = request.num_output_tokens == 0
                    if is_prefill:
                        if isinstance(self.connector, MultiConnector):
                            for c in self.connector._connectors:
                                if hasattr(c, "connector") and hasattr(
                                    c.connector, "commit"
                                ):
                                    c.connector.commit(
                                        model_runner_output.finished_dumping.get(
                                            req_id, []
                                        ),
                                        True,
                                    )
                        else:
                            self.connector.connector.commit(
                                model_runner_output.finished_dumping.get(req_id, []),
                                True,
                            )

                # Append generated tokens and check for stop. Note that if
                # a request is still being prefilled, we expect the model runner
                # to return empty token ids for the request.
                for num_new, output_token_id in enumerate(new_token_ids, 1):
                    request.append_output_token_ids(output_token_id)

                    # Check for stop and update request state.
                    # This must be called before we make the EngineCoreOutput.
                    stopped = check_stop(request, self.max_model_len)
                    if stopped:
                        kv_transfer_params = self._free_request(request)
                        del new_token_ids[num_new:]  # Trim new tokens if needed.
                        break

                pooler_output = None
                if pooler_outputs:
                    pooler_output = pooler_outputs[req_index]
                    stopped = check_stop(request, self.max_model_len, pooler_output)
                    if stopped:
                        kv_transfer_params = self._free_request(request)

                # Extract sample logprobs if needed.
                if (
                    request.sampling_params is not None
                    and request.sampling_params.logprobs is not None
                    and logprobs
                ):
                    # NOTE: once we support N tokens per step (spec decode),
                    # the outer lists can be of length > 1.
                    new_logprobs = logprobs.slice(req_index, req_index + 1)

                if new_token_ids and self.structured_output_manager.should_advance(
                    request
                ):
                    # NOTE: structured_output_request
                    # should not be None if use_structured_output, we have
                    # check above, so safe to ignore type warning
                    request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                        req_id, new_token_ids
                    )

                # spec_token_ids comes from the model runner output
                if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                    request.num_nans_in_logits = num_nans_in_logits[req_id]

                # Add newly generated spec token ids to the request.
                if spec_token_ids is not None:
                    if self.structured_output_manager.should_advance(request):
                        metadata = request.structured_output_request
                        # Needs to happen after new_token_ids are accepted.
                        request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                            spec_token_ids[req_index]
                        )
                    else:
                        request.spec_token_ids = spec_token_ids[req_index]

                # Get prompt logprobs for this request.
                prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
                if new_token_ids or pooler_output is not None or kv_transfer_params:

                    # Add EngineCoreOutput for this Request.
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=new_token_ids,
                            finish_reason=request.get_finished_reason(),
                            new_logprobs=new_logprobs,
                            new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                            pooling_output=pooler_output,
                            stop_reason=request.stop_reason,
                            events=request.take_events(),
                            kv_transfer_params=kv_transfer_params,
                            num_cached_tokens=request.num_cached_tokens,
                        )
                    )

                else:
                    # Invariant: EngineCore returns no partial prefill outputs.
                    assert not prompt_logprobs_tensors

                if not stopped:
                    new_running.append(request)
            self.running = new_running

            # KV Connector: update state for finished KV Transfers.
            self._update_from_kv_xfer_finished(model_runner_output)

            # Create EngineCoreOutputs for all clients that have requests with
            # outputs in this step.
            engine_core_outputs = {
                client_index: EngineCoreOutputs(outputs=outs)
                for client_index, outs in outputs.items()
            }

            finished_req_ids = self.finished_req_ids_dict
            if finished_req_ids:
                # Include ids of requests that finished since last outputs
                # were sent.
                for client_index, finished_set in finished_req_ids.items():
                    # Set finished request set in EngineCoreOutputs for this client.
                    if (eco := engine_core_outputs.get(client_index)) is not None:
                        eco.finished_requests = finished_set
                    else:
                        engine_core_outputs[client_index] = EngineCoreOutputs(
                            finished_requests=finished_set
                        )
                finished_req_ids.clear()

            if engine_core_outputs:
                # Return stats to only one of the front-ends.
                next(iter(engine_core_outputs.values())).scheduler_stats = (
                    self.make_stats(spec_decoding_stats)
                )

            return engine_core_outputs

        Scheduler.update_from_output = update_from_output
    except ImportError:
        logger.warning(
            "Could not patch Scheduler.update_from_output - module not found"
        )


# ==================== vllm/v1/worker/gpu_model_runner.py ====================
def _patch_gpu_model_runner_finished_dumping() -> None:
    """Patch GPUModelRunner.maybe_wait_for_kv_save to return finished_dumping."""
    try:
        from typing import TYPE_CHECKING, Optional, Union

        import torch
        import vllm.envs as envs
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.forward_context import set_forward_context
        from vllm.sequence import IntermediateTensors
        from vllm.utils import round_up
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

        if TYPE_CHECKING:
            from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        @staticmethod
        def maybe_wait_for_kv_save() -> Optional[dict[str, list[str]]]:
            if has_kv_transfer_group():
                return get_kv_transfer_group().wait_for_save()
            return None

        GPUModelRunner.maybe_wait_for_kv_save = maybe_wait_for_kv_save

        @torch.inference_mode()
        def execute_model(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> Union[ModelRunnerOutput, IntermediateTensors]:
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT

                return self.kv_connector_no_forward(scheduler_output)

            # Prepare the decoder inputs.
            (
                attn_metadata,
                attention_cuda_graphs,
                logits_indices,
                spec_decode_metadata,
                num_scheduled_tokens_np,
            ) = self._prepare_inputs(scheduler_output)
            num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            if (
                self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
            ):
                # Use piecewise CUDA graphs.
                # Add padding to the batch size.
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    num_scheduled_tokens
                )
            else:
                # Eager mode.
                # Pad tokens to multiple of tensor_parallel_size when
                # enabled collective fusion for SP
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                if (
                    self.compilation_config.pass_config.enable_sequence_parallelism
                    and tp_size > 1
                ):
                    num_input_tokens = round_up(num_scheduled_tokens, tp_size)
                else:
                    num_input_tokens = num_scheduled_tokens

            # Padding for DP
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
            num_input_tokens += num_pad

            # _prepare_inputs may reorder the batch, so we must gather multi
            # modal outputs after that to ensure the correct order
            if self.is_multimodal_model:
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)
                mm_embeds = self._gather_mm_embeddings(scheduler_output)
            else:
                mm_embeds = []

            if self.is_multimodal_model and get_pp_group().is_first_rank:
                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids[:num_scheduled_tokens]
                if mm_embeds:
                    inputs_embeds = self.model.get_input_embeddings(
                        input_ids, mm_embeds
                    )
                else:
                    inputs_embeds = self.model.get_input_embeddings(input_ids)
                # TODO(woosuk): Avoid the copy. Optimize.
                self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
                input_ids = None
            else:
                # For text-only models, we use token ids as input.
                # While it is possible to use embeddings as input just like the
                # multimodal models, it is not desirable for performance since
                # then the embedding layer is not included in the CUDA graph.
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_input_tokens]
            else:
                positions = self.positions[:num_input_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_input_tokens, intermediate_tensors, True
                )

            # Some attention backends only support CUDA Graphs in pure decode.
            # If attention doesn't support CUDA Graphs for this batch, but we
            # compiled with full CUDA graphs, we have to skip them entirely.
            skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

            # Run the model.
            # Use persistent buffers for CUDA graphs.
            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
            ):
                self.maybe_setup_kv_connector(scheduler_output)

                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

                finished_dumping = self.maybe_wait_for_kv_save()
                finished_sending, finished_recving = self.get_finished_kv_transfers(
                    scheduler_output
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = (
                self.parallel_config.distributed_executor_backend == "external_launcher"
                and len(get_pp_group().ranks) > 0
            )
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group()
                )
                logits = None
            else:
                if self.input_batch.pooling_params:
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        finished_sending,
                        finished_recving,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
            if broadcast_pp_output:
                model_output_broadcast_data = (
                    {
                        "logits": logits.contiguous(),
                    }
                    if logits is not None
                    else {}
                )
                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                assert logits is not None
                bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids

            num_nans_in_logits = {}
            if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
                num_nans_in_logits = self._get_nans_in_logits(logits)

            # TODO(woosuk): The following loop can be slow since it iterates over
            # the requests one by one. Optimize.
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (
                    req_state.num_computed_tokens
                    + scheduler_output.num_scheduled_tokens[req_id]
                )
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token for partial prefills.
                    # Rewind the generator state as if the token was not sampled.
                    # This relies on cuda-specific torch-internal impl details
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(generator.get_offset() - 4)
                    # Record the index of the request that should not be sampled,
                    # so that we could clear the sampled tokens before returning.
                    discard_sampled_tokens_req_indices.append(i)

            # NOTE: GPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = (
                logprobs_tensors.tolists() if logprobs_tensors is not None else None
            )

            # Compute prompt logprobs if needed.
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[:num_scheduled_tokens],
                scheduler_output,
            )

            # Get the valid generated tokens.
            sampled_token_ids = sampler_output.sampled_token_ids
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = sampled_token_ids.tolist()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            # Cache the sampled tokens in the model runner, so that the scheduler
            # doesn't need to send them back.
            # NOTE(woosuk): As an exception, when using PP, the scheduler sends
            # the sampled tokens back, because there's no direct communication
            # between the first-stage worker and the last-stage worker.
            for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
                if not sampled_ids:
                    continue

                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(sampled_ids)
                assert end_idx <= self.max_model_len, (
                    "Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.max_model_len}"
                )

                self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
                self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                self.input_batch.num_tokens[req_idx] = end_idx
                req_id = self.input_batch.req_ids[req_idx]
                req_state = self.requests[req_id]
                req_state.output_token_ids.extend(sampled_ids)

            if not self.speculative_config:
                # Speculative decoding is not enabled.
                spec_token_ids = None
            else:
                spec_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    valid_sampled_token_ids,
                    sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    attn_metadata,
                )

            # Clear KVConnector state after all KVs are generated.
            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

            self.eplb_step()

            return ModelRunnerOutput(
                req_ids=self.input_batch.req_ids,
                req_id_to_index=self.input_batch.req_id_to_index,
                sampled_token_ids=valid_sampled_token_ids,
                spec_token_ids=spec_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                finished_dumping=finished_dumping,
                num_nans_in_logits=num_nans_in_logits,
            )

        # Also patch the execute_model method to capture finished_dumping
        GPUModelRunner.execute_model = execute_model
    except ImportError:
        logger.warning("Could not patch GPUModelRunner - module not found")


def _apply_pc_patch() -> None:
    """Apply vllm-adapt-pc.patch changes."""
    try:
        # Patch 1: request.py - add succeed_dumped_blocks field
        _patch_request_succeed_dumped_blocks()

        # Patch 2: multi_connector.py - wait_for_save return value
        _patch_multi_connector_wait_for_save()

        # Patch 3: outputs.py - add finished_dumping field
        _patch_outputs_finished_dumping()

        # Patch 4: gpu_model_runner.py - return finished_dumping
        _patch_gpu_model_runner_finished_dumping()

        # Patch 5: scheduler.py - add finished_dumping handling
        _patch_scheduler_update_from_output()

        logger.debug("Applied PC patch")
    except Exception as e:
        logger.error(f"Failed to apply PC patch: {e}", exc_info=True)
        raise
