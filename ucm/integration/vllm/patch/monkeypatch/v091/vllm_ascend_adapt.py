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
# Adapted from lmcache/lmcache/integration/vllm/vllm_v1_adapter.py
#

from __future__ import annotations

from ucm.logger import init_logger

logger = init_logger(__name__)


def _apply_ascend_patch() -> None:
    """Apply patches for vLLM-Ascend 0.9.1."""
    logger.info("Applying patch for vLLM-Ascend0.9.1...")
    _patch_attention_v1()
    _patch_model_runner_v1()

# ========================= vllm_ascend/attention/attention_v1.py =========================
def _patch_attention_v1() -> None:
    """Patch attention_v1.py for vLLM-Ascend0.9.1."""
    logger.info("Patching attention_v1.py for vLLM-Ascend0.9.1...")
    try:
        from typing import List

        import torch
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
            is_v1_kv_transfer_group,
        )
        from vllm.forward_context import ForwardContext, get_forward_context
        from vllm_ascend.attention import attention_v1

        def unified_ascend_attention_with_output(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output: torch.Tensor,
            layer_name: str,
        ) -> None:
            wait_for_kv_layer_from_connector(layer_name)
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            self = forward_context.no_compile_layers[layer_name]
            kv_cache = self.kv_cache[forward_context.virtual_engine]
            self.impl.forward(self,
                            query,
                            key,
                            value,
                            kv_cache,
                            attn_metadata,
                            output,
                            trace_flag=False)
            maybe_save_kv_layer_to_connector(layer_name, kv_cache)
            return
        attention_v1.unified_ascend_attention_with_output = unified_ascend_attention_with_output

        def wait_for_kv_layer_from_connector(layer_name: str):
            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            connector.wait_for_layer_load(layer_name)
        attention_v1.wait_for_kv_layer_from_connector = wait_for_kv_layer_from_connector

        def maybe_save_kv_layer_to_connector(
            layer_name: str,
            kv_cache_layer: List[torch.Tensor],
        ):
            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            connector.save_kv_layer(layer_name, kv_cache_layer,
                                    attn_metadata)
        attention_v1.maybe_save_kv_layer_to_connector = maybe_save_kv_layer_to_connector

    except ImportError as e:
        logger.error(f"Failed to patch attention_v1.py: {e}", exc_info=True)
        raise

# ========================= vllm_ascend/worker/model_runner_v1.py =========================
def _patch_model_runner_v1() -> None:
    """Patch model_runner_v1.py for vLLM-Ascend0.9.1."""
    logger.info("Patching model_runner_v1.py for vLLM-Ascend0.9.1...")
    try:
        from typing import TYPE_CHECKING, Optional, Union

        import numpy as np
        import torch
        from vllm.logger import logger
        from vllm.sequence import IntermediateTensors
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionState,
        )
        from vllm_ascend.utils import (
            ProfileExecuteDuration,
        )


        if TYPE_CHECKING:
            from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
        import vllm_ascend.envs as envs_ascend
        from vllm_ascend.distributed.utils import is_lmhead_tp
        import torch.nn as nn
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context


        def _process_reqs(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> tuple[SpecDecodeMetadata, torch.Tensor, SpecDecodeMetadata,
                torch.Tensor, int, torch.Tensor, Optional[set[str]],
                Optional[set[str]], Optional[dict[str, list[str]]]]:
            # Check input valid
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            assert total_num_scheduled_tokens > 0
            num_reqs = self.input_batch.num_reqs
            assert num_reqs > 0
            if (self.use_aclgraph and
                    total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]):
                # Add padding to the batch size.
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    total_num_scheduled_tokens)
            else:
                # Eager mode.
                num_input_tokens = total_num_scheduled_tokens

            modified_batch = self.attn_metadata_builder.reorder_batch(
                self.input_batch, scheduler_output)
            if modified_batch:
                self.input_batch.refresh_sampling_metadata()

            # OPTIMIZATION: Start copying the block table first.
            # This way, we can overlap the copy with the following CPU operations.
            self.input_batch.block_table.commit(num_reqs)

            # Get the number of scheduled tokens for each request.
            # TODO: The Python loop can be slow. Optimize.
            num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
            num_valid_tokens = np.empty(num_reqs, dtype=np.int32)
            max_num_scheduled_tokens = 0
            for i, req_id in enumerate(self.input_batch.req_ids):
                num_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_scheduled_tokens[i] = num_tokens
                num_valid_tokens[i] = num_tokens - \
                    len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
                max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                            num_tokens)

            # Hot-Swap lora model
            if self.lora_config:
                self.set_active_loras(self.input_batch, num_scheduled_tokens)

            # Prepare positions
            req_indices = np.repeat(self.arange_np[:num_reqs],
                                    num_scheduled_tokens)
            cu_num_tokens = np.cumsum(num_scheduled_tokens)
            cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                        num_scheduled_tokens)
            sample_indices = cu_num_tokens - 1
            sample_indices = torch.from_numpy(sample_indices).to(self.device,
                                                                non_blocking=True)
            arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

            positions_np = self.positions_np[:total_num_scheduled_tokens]
            np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np)

            # Calculate M-RoPE positions.
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._calc_mrope_positions(scheduler_output)

            if self.uses_mrope:
                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                    self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                    non_blocking=True)

            self.positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
            self.positions[:num_input_tokens].copy_(
                self.positions_cpu[:num_input_tokens], non_blocking=True)
            positions_cpu = self.positions_cpu[:num_input_tokens]
            positions = self.positions[:num_input_tokens]
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

            self.seq_lens_np[:num_reqs] = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                num_scheduled_tokens)
            seq_lens_cpu = self.seq_lens_cpu[:num_reqs]

            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                positions_np // self.block_size)

            block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
            block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
            block_offsets = positions_np % self.block_size
            np.add(block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping_np[:total_num_scheduled_tokens])

            ascend_config = get_ascend_config()
            use_spec_decode = len(
                scheduler_output.scheduled_spec_decode_tokens) > 0
            if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
                attn_state = AscendAttentionState.PrefillNoCache
            # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
            elif np.all(num_scheduled_tokens == 1):
                attn_state = AscendAttentionState.DecodeOnly
                if self.speculative_config and self.speculative_config.method == 'deepseek_mtp':
                    # support deepseek mtp spec decode in disaggregated-prefill scenario
                    attn_state = AscendAttentionState.SpecDecoding
            # Speculative decoding.
            elif np.all(num_valid_tokens == 1):
                attn_state = AscendAttentionState.SpecDecoding
            # splitfuse
            elif not ascend_config.ascend_scheduler_config.enabled or self.chunked_prefill_enabled:
                attn_state = AscendAttentionState.ChunkedPrefill
            else:
                attn_state = AscendAttentionState.PrefillCacheHit

            # NOTE: when use ring_mla, attn_mask don't need to generate here.
            if not self.vllm_config.model_config.use_mla:
                attn_mask = self._make_attention_mask(seq_lens=seq_lens_cpu,
                                                    position=positions_cpu,
                                                    attn_state=attn_state)
                self.attn_mask = attn_mask
            self.attn_state = attn_state  # type: ignore

            extra_builder_kwargs = {}

            self.query_start_loc_np[0] = 0
            self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
            self.query_start_loc[:num_reqs + 1].copy_(
                self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
            self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                        non_blocking=True)
            self.slot_mapping[:total_num_scheduled_tokens].copy_(
                self.slot_mapping_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

            # Fill unused with -1. Needed for reshape_and_cache
            self.slot_mapping[total_num_scheduled_tokens:].fill_(-1)
            self.seq_lens[num_reqs:].fill_(0)
            self.query_start_loc[num_reqs + 1:].fill_(-1)

            # Use host tensor, other wise error: tensor.hostData is null
            self.seq_lens_list = self.seq_lens_np.tolist()[:num_input_tokens]
            with_prefill = attn_state not in [
                AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
            ]

            is_only_prefill = bool(np.all(num_valid_tokens != 1))

            enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(),
                                                attn_state,
                                                total_num_scheduled_tokens)

            maybe_padded_num_tokens = total_num_scheduled_tokens
            if self.torchair_graph_enabled and not with_prefill:
                maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                    total_num_scheduled_tokens)
            (padded_num_tokens_across_dp, num_tokens_across_dp, with_prefill,
            enable_dbo) = self._get_forward_metadata_across_dp(
                maybe_padded_num_tokens, total_num_scheduled_tokens, with_prefill,
                enable_dbo)

            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
                seq_lens=self.seq_lens[:num_reqs],
                seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                block_table_tensor=self.input_batch.block_table[0].
                get_device_tensor(),
                slot_mapping_cpu=self.
                slot_mapping_cpu[:total_num_scheduled_tokens],
                positions=self.positions[:num_input_tokens],
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,  # type: ignore
                decode_token_per_req=self.decode_token_per_req,
                max_num_blocks_per_req=self.max_num_blocks_per_req,
                enable_dbo_across_dp=enable_dbo,
                is_only_prefill=is_only_prefill,
            )

            # TODO(zzzzwwjj): this code need to refactor afterwards.
            self.with_prefill = with_prefill
            # Add num_token_pad_size and num_reqs_pad_size here for torchair graph mode
            if self.torchair_graph_enabled and not with_prefill:
                num_token_pad_size = padded_num_tokens_across_dp - total_num_scheduled_tokens
                num_reqs_pad_size = (
                    padded_num_tokens_across_dp // self.decode_token_per_req -
                    num_reqs)
                assert num_token_pad_size >= 0 and num_reqs_pad_size >= 0

                extra_builder_kwargs['num_token_pad_size'] = num_token_pad_size
                extra_builder_kwargs['num_reqs_pad_size'] = num_reqs_pad_size
                self.num_reqs_pad_size = num_reqs_pad_size
                self.num_token_pad_size = num_token_pad_size
            self.extra_builder_kwargs = extra_builder_kwargs
            self.num_tokens_across_dp = num_tokens_across_dp

            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                common_attn_metadata=common_attn_metadata,
                **extra_builder_kwargs,
            )
            attn_metadata.num_input_tokens = padded_num_tokens_across_dp

            # Prepare input_ids
            token_indices = (positions_np +
                            req_indices * self.input_batch.token_ids_cpu.shape[1])
            torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                            0,
                            torch.from_numpy(token_indices),
                            out=self.input_ids_cpu[:total_num_scheduled_tokens])
            # Copy the tensors to the NPU.
            self.input_ids[:total_num_scheduled_tokens].copy_(
                self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)

            # _prepare_inputs may reorder the batch, so we must gather multi
            # modal outputs after that to ensure the correct order
            if self.is_multimodal_model:
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)
                mm_embeds = self._gather_mm_embeddings(scheduler_output)
            else:
                mm_embeds = []

            if self.is_multimodal_model:
                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids[:total_num_scheduled_tokens]
                if mm_embeds:
                    inputs_embeds = self.model.get_input_embeddings(
                        input_ids, mm_embeds)
                else:
                    inputs_embeds = self.model.get_input_embeddings(input_ids)
                # TODO(woosuk): Avoid the copy. Optimize.
                self.inputs_embeds[:total_num_scheduled_tokens].copy_(
                    inputs_embeds)
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
                input_ids = None
            else:
                # For text-only models, we use token ids as input.
                # While it is possible to use embeddings as input just like the
                # multimodal models, it is not desirable for performance since
                # then the embedding layer is not included in the ACL Graph.
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_input_tokens]

            if self.torchair_graph_enabled and not with_prefill:
                input_ids = self.input_ids[:padded_num_tokens_across_dp]
                positions = self.positions[:padded_num_tokens_across_dp]

            # Run forward pass
            finished_dumping = None
            # TODO(zzzzwwjj): check param `num_tokens_across_dp` later.
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=padded_num_tokens_across_dp,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=with_prefill,
                    num_actual_tokens=total_num_scheduled_tokens):
                with ProfileExecuteDuration().capture_async("forward"):
                    self.maybe_setup_kv_connector(scheduler_output)
                    model_kwargs = {}
                    if self.torchair_graph_enabled:
                        model_kwargs["kv_caches"] = self.kv_caches
                        model_kwargs["attn_metadata"] = attn_metadata
                    if envs_ascend.VLLM_ASCEND_ENABLE_DBO:
                        if with_prefill:
                            model_kwargs["graph_enable"] = False  # type: ignore
                        else:
                            model_kwargs["graph_enable"] = True  # type: ignore
                    if self.torchair_graph_enabled and not with_prefill:
                        compiled_model = self._get_torchair_lazy_compiled_model(
                            padded_num_tokens_across_dp)
                        hidden_states = compiled_model(
                            input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            **model_kwargs)
                    else:
                        assert self.model is not None
                        hidden_states = self.model(
                            input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            **model_kwargs)

            finished_dumping = self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)
            use_spec_decode = len(
                scheduler_output.scheduled_spec_decode_tokens) > 0
            if not use_spec_decode:
                # NOTE(woosuk): Due to chunked prefills, the batch may contain
                # partial requests. While we should not sample any token
                # from these partial requests, we do so for simplicity.
                # We will ignore the sampled tokens from the partial requests.
                # TODO: Support prompt logprobs.
                spec_decode_metadata = None
            else:
                # Get the number of draft tokens for each request.
                # Iterate over the dictionary rather than all requests since not all
                # requests have draft tokens.
                num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
                for req_id, draft_token_ids in (
                        scheduler_output.scheduled_spec_decode_tokens.items()):
                    req_idx = self.input_batch.req_id_to_index[req_id]
                    num_draft_tokens[req_idx] = len(draft_token_ids)

                spec_decode_metadata = self._calc_spec_decode_metadata(
                    num_draft_tokens, cu_num_tokens)
                sample_indices = spec_decode_metadata.logits_indices

            if is_lmhead_tp():
                if not with_prefill:
                    padded_num_indices = padded_num_tokens_across_dp
                else:
                    padded_num_indices = self.max_num_reqs
                sample_indices = nn.functional.pad(
                    sample_indices,
                    (0, padded_num_indices - sample_indices.shape[0]))

            return (attn_metadata, hidden_states, spec_decode_metadata, positions,
                    total_num_scheduled_tokens, sample_indices, finished_sending,
                    finished_recving, finished_dumping)
        NPUModelRunner._process_reqs = _process_reqs

        @torch.inference_mode()
        def execute_model(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> Union[ModelRunnerOutput, torch.Tensor]:
            with ProfileExecuteDuration().capture_async(
                    "prepare input and forward"):
                self._update_states(scheduler_output)
                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        logger.debug(
                            "skip this step for we receive the data from remote disaggregate prefill node"
                        )
                        # Return empty ModelRunnerOuptut if there's no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output)

                if self.dynamic_eplb:
                    self.eplb_updator.forward_before()

                (attn_metadata, hidden_states, spec_decode_metadata, positions,
                num_scheduled_tokens, sample_indices, finished_sending,
                finished_recving, finished_dumping) = (self._process_reqs(scheduler_output,
                                                        intermediate_tensors))

                if self.dynamic_eplb:
                    self.eplb_updator.take_update_info_from_eplb_process()

            with ProfileExecuteDuration().capture_async("post process"):
                logits = self.model.compute_logits(hidden_states[sample_indices],
                                                None)

                # Apply structured output bitmasks if present
                if scheduler_output.grammar_bitmask is not None:
                    logits = self.apply_grammar_bitmask(scheduler_output, logits)

                # Sample the next token and get logprobs if needed.
                sampling_metadata = self.input_batch.sampling_metadata
                if spec_decode_metadata is None:
                    if is_lmhead_tp():
                        logits = logits[:self.input_batch.num_reqs]

                    sampler_output = self.sampler(
                        logits=logits,
                        sampling_metadata=sampling_metadata,
                    )
                else:
                    if is_lmhead_tp():
                        logits = logits[:len(spec_decode_metadata.logits_indices)]

                    # When indexing with a tensor (bonus_logits_indices), PyTorch
                    # creates a new tensor with separate storage from the original
                    # logits tensor. This means any in-place operations on bonus_logits
                    # won't affect the original logits tensor.
                    bonus_logits = logits[
                        spec_decode_metadata.bonus_logits_indices]
                    sampler_output = self.sampler(
                        logits=bonus_logits,
                        sampling_metadata=sampling_metadata,
                    )
                    bonus_token_ids = sampler_output.sampled_token_ids

                    # Just like `bonus_logits`, `target_logits` is a new tensor with
                    # separate storage from the original `logits` tensor. Therefore,
                    # it is safe to update `target_logits` in place.
                    target_logits = logits[
                        spec_decode_metadata.target_logits_indices]
                    output_token_ids = self.rejection_sampler(
                        spec_decode_metadata,
                        None,  # draft_probs
                        target_logits,
                        bonus_token_ids,
                        sampling_metadata,
                    )
                    sampler_output.sampled_token_ids = output_token_ids

                # TODO(woosuk): The following loop can be slow since it iterates over
                # the requests one by one. Optimize.
                discard_sampled_tokens_req_indices = []
                for i, req_id in enumerate(self.input_batch.req_ids):
                    req_state = self.requests[req_id]
                    seq_len = (req_state.num_computed_tokens +
                            scheduler_output.num_scheduled_tokens[req_id])
                    if seq_len < req_state.num_tokens:
                        # Ignore the sampled token.
                        # Rewind the generator state as if the token was not sampled.
                        generator = self.input_batch.generators.get(i)
                        if generator is not None:
                            generator.set_offset(generator.get_offset() - 4)
                        discard_sampled_tokens_req_indices.append(i)

                # NOTE: NPU -> CPU Sync happens here.
                # Move as many CPU operations as possible before this sync point.
                logprobs_tensors = sampler_output.logprobs_tensors
                logprobs_lists = logprobs_tensors.tolists() \
                    if logprobs_tensors is not None else None

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

                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[i].clear()

                spec_token_ids = self._get_spec_token_ids(
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    positions,
                    num_scheduled_tokens,
                    hidden_states,
                    attn_metadata,
                )
                if has_kv_transfer_group():
                    get_kv_transfer_group().clear_connector_metadata()

                model_runner_output = ModelRunnerOutput(
                    req_ids=self.input_batch.req_ids,
                    req_id_to_index=self.input_batch.req_id_to_index,
                    sampled_token_ids=valid_sampled_token_ids,
                    spec_token_ids=spec_token_ids,
                    logprobs=logprobs_lists,
                    prompt_logprobs_dict={},
                    finished_sending=finished_sending,
                    finished_recving=finished_recving,
                    finished_dumping=finished_dumping
                )

            durations = ProfileExecuteDuration().pop_captured_sync()
            if durations:
                dr_str = [
                    f"[{tag}]:{duration:.2f}ms"
                    for tag, duration in durations.items()
                ]
                captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
                logger.info("Profile execute duration [%s]:%s", captured_name,
                            " ".join(dr_str))

            if self.dynamic_eplb:
                self.eplb_updator.forward_end()

            return model_runner_output
        NPUModelRunner.execute_model = execute_model

        @staticmethod
        def maybe_wait_for_kv_save() -> None:
            if has_kv_transfer_group():
                return get_kv_transfer_group().wait_for_save()
        NPUModelRunner.maybe_wait_for_kv_save = maybe_wait_for_kv_save

    except ImportError as e:
        logger.error(f"Failed to patch model_runner_v1.py: {e}", exc_info=True)
        raise