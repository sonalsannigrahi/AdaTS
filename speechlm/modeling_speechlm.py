# coding=utf-8
"""Speech LM architecture with token compression"""

from typing import Optional, Tuple, Union, Dict
import torch
from torch import nn
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_speechlm import SpeechLMConfig

from ..wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertAdapterLayer


logger = logging.get_logger(__name__)

class SpeechLMPreTrainedModel(PreTrainedModel, GenerationMixin):
    base_class_prefix = "model"
    _skip_keys_device_placement = ["past_key_values"]
    _no_split_modules = [
        "LlamaDecoderLayer",
        "SpeechLMPreAdapter",
        "Wav2Vec2BertAdapterLayer",
        "Wav2Vec2BertEncoderLayer",
    ]


class SpeechLMPreAdapter(nn.Module):
    def __init__(self, config):

        super().__init__()
        # feature dim might need to be down-projected
        self.proj = nn.Linear(
            config.encoder.feature_projection_input_dim,
            config.encoder.output_hidden_size,
        )
        self.proj_layer_norm = nn.LayerNorm(
            config.encoder.output_hidden_size, eps=config.encoder.layer_norm_eps
        )
        self.layers = nn.ModuleList(
            Wav2Vec2BertAdapterLayer(config.encoder)
            for _ in range(config.num_pre_adapter_layers)
        )
        self.layerdrop = config.encoder.layerdrop

        self.kernel_size = config.encoder.adapter_kernel_size
        self.stride = config.encoder.adapter_stride
        self.out_proj = nn.Linear(
            config.encoder.output_hidden_size,
            config.encoder.feature_projection_input_dim,
        )

    def _compute_sub_sample_lengths_from_attention_mask(self, seq_lens):
        if seq_lens is None:
            return seq_lens
        pad = self.kernel_size // 2
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1
        return seq_lens.floor()

    def forward(self, hidden_states, attention_mask=None):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        sub_sampled_lengths = None
        if attention_mask is not None:
            sub_sampled_lengths = (
                attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
            ).to(hidden_states.device)

        for layer in self.layers:
            layerdrop_prob = torch.rand([])
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(
                sub_sampled_lengths
            )
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sub_sampled_lengths=sub_sampled_lengths,
                )

        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class SpeechLMForConditionalGeneration(SpeechLMPreTrainedModel):
    r"""
    [`SpeechEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    one of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    config_class = SpeechLMConfig
    base_model_prefix = "speech_lm"
    main_input_name = "input_ids"

    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    # TODO: this gets ignored by the trainer, maybe it goes in the config class
    loss_type: str = "ForCausalLM"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError(
                "Either a configuration or an encoder and a decoder has to be provided."
            )
        if config is None:
            config = SpeechLMForConditionalGeneration.from_encoder_decoder_configs(
                encoder.config, decoder.config
            )
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"Config: {config} has to be of type {self.config_class}"
                )

        self.loss_type = "ForCausalLM"

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.config.encoder._attn_implementation = (
            self.encoder.config._attn_implementation
        )
        self.config.decoder._attn_implementation = (
            self.decoder.config._attn_implementation
        )
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # get encoder output hidden size
        self.encoder_output_dim = getattr(
            config.encoder,
            "output_hidden_size",
            config.encoder.hidden_size,
        )

        ####################################
        # MODALITY AND LENGTH ADAPTER
        # TODO: be back at this with better strategies
        ####################################
        if self.encoder_output_dim != self.decoder.config.hidden_size:
            logger.info("Adding encoder to decoder projection layer")
            self.enc_to_dec_proj = nn.Linear(
                self.encoder.config.hidden_size, self.decoder.config.hidden_size
            )

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        if config.add_pre_adapter:
            self.pre_adapter = SpeechLMPreAdapter(self.config)

    def get_encoder(self):
        encoder = self.encoder
        if hasattr(encoder, 'encoder'):
            encoder = encoder.encoder  # Unwrap ChunkedAudioEncoder
        return encoder
    
    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def get_compression_statistics(self, audio_duration_ms: Optional[float] = None):
        """
        Get compression statistics and optionally calculate time per token.
        
        Args:
            audio_duration_ms: Duration of audio in milliseconds (optional)
            
        Returns:
            Dictionary with compression statistics including avg time per token if duration provided
        """
        if not hasattr(self, 'compression_stats_history') or len(self.compression_stats_history) == 0:
            return {
                'avg_compression_ratio': 1.0,
                'total_original_tokens': 0,
                'total_compressed_tokens': 0,
            }
        
        # Aggregate statistics from history
        total_original = sum(s['original_tokens'] for s in self.compression_stats_history)
        total_compressed = sum(s['compressed_tokens'] for s in self.compression_stats_history)
        avg_ratio = sum(s['compression_ratio'] for s in self.compression_stats_history) / len(self.compression_stats_history)
        
        stats = {
            'avg_compression_ratio': avg_ratio,
            'total_original_tokens': total_original,
            'total_compressed_tokens': total_compressed,
            'num_batches': len(self.compression_stats_history),
        }
        
        # Calculate time per token if audio duration provided
        if audio_duration_ms is not None and total_compressed > 0:
            stats['avg_ms_per_compressed_token'] = audio_duration_ms / total_compressed
            if total_original > 0:
                stats['avg_ms_per_original_token'] = audio_duration_ms / total_original
        
        return stats

    def reset_compression_statistics(self):
        """Reset compression statistics history."""
        if hasattr(self, 'compression_stats_history'):
            self.compression_stats_history = []

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder of the speech encoder so
        that its parameters will not be updated during training.
        We freeze all parameters but the adapter.
        """
        # Handle both regular encoder and chunked encoder
        encoder = self.encoder
        if hasattr(encoder, 'encoder'):
            encoder = encoder.encoder
        
        for name, param in encoder.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    def freeze_decoder(self):
        """
        Calling this function will disable the gradient computation for the decoder so that its parameters will not be
        updated during training.
        """
        for param in self.decoder.parameters():
            param.requires_grad = False

    def compress_tokens_by_similarity(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        similarity_threshold: float = 0.6,
        weighted: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compress tokens by merging adjacent tokens with cosine similarity above threshold.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            similarity_threshold: Threshold for merging tokens (0.0 to 1.0)
            weighted: If True, merge each group with weights (1 - sim(a_i, a_{i-1})),
                    normalised to sum to 1. If False, use a plain mean.
            
        Returns:
            compressed_hidden_states: Compressed tensor
            compressed_attention_mask: Updated attention mask
            compression_stats: Dictionary with compression statistics
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Track compression statistics
        total_original_tokens = 0
        total_compressed_tokens = 0
        
        # Process each sample in the batch independently
        compressed_batch = []
        compressed_masks = []
        
        for b in range(batch_size):
            sample_hidden = hidden_states[b]  # (seq_len, hidden_dim)
            
            # Get valid tokens based on attention mask
            if attention_mask is not None:
                valid_mask = attention_mask[b].bool()
                valid_length = valid_mask.sum().item()
            else:
                valid_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
                valid_length = seq_len
            
            if valid_length <= 1:
                compressed_batch.append(sample_hidden)
                compressed_masks.append(valid_mask if attention_mask is not None else torch.ones(seq_len, device=hidden_states.device))
                total_original_tokens += valid_length
                total_compressed_tokens += valid_length
                continue
            
            valid_tokens = sample_hidden[valid_mask]  
            valid_tokens = valid_tokens.detach()
            
            normalized = F.normalize(valid_tokens, p=2, dim=-1)
            similarities = torch.sum(normalized[:-1] * normalized[1:], dim=-1)  # (valid_length-1,)

            groups = []
            current_group = [0]  # Start with first token
            
            for i, sim in enumerate(similarities):
                if sim >= similarity_threshold:
                    current_group.append(i + 1)
                else:
                    groups.append(current_group)
                    current_group = [i + 1]
            
            groups.append(current_group)
            compressed_tokens = []
            for group in groups:
                group_tokens = valid_tokens[group]  # (group_size, hidden_dim)
                merged_token = group_tokens.mean(dim=0)  # (hidden_dim,)
                compressed_tokens.append(merged_token)
            
            if weighted:
                normalized = F.normalize(valid_tokens, p=2, dim=-1)
                similarities = torch.sum(normalized[:-1] * normalized[1:], dim=-1)  # (valid_length-1,)

                # w_i = 1 - sim(a_i, a_{i-1}); the first token has no predecessor, so sim is 0 (weight 1)
                sim_prev = torch.cat([similarities.new_zeros(1), similarities])     # (valid_length,)
                weights = (1.0 - sim_prev).clamp(min=0.0) + 1e-6                    # (valid_length,)

                compressed_tokens = []
                for group in groups:
                    group_tokens = valid_tokens[group]  # (group_size, hidden_dim)

                    if weighted and len(group) > 1:
                        # Formula 1: normalised weighted average within the group
                        w = weights[group].unsqueeze(-1).to(group_tokens.dtype)  # (group_size, 1)
                        merged_token = (w * group_tokens).sum(dim=0) / w.sum(dim=0)
                    else:
                        merged_token = group_tokens.mean(dim=0)  # (hidden_dim,)

                    compressed_tokens.append(merged_token)

            # Stack compressed tokens
            compressed_sample = torch.stack(compressed_tokens)  # (compressed_len, hidden_dim)
            
            # Update statistics
            total_original_tokens += valid_length
            total_compressed_tokens += len(compressed_tokens)
            
            # Pad to original sequence length if needed
            compressed_len = compressed_sample.shape[0]
            if compressed_len < seq_len:
                padding = torch.zeros(
                    seq_len - compressed_len, 
                    hidden_dim, 
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                compressed_sample = torch.cat([compressed_sample, padding], dim=0)
                
                # Update mask
                new_mask = torch.zeros(seq_len, dtype=torch.float32, device=hidden_states.device)
                new_mask[:compressed_len] = 1.0
            else:
                new_mask = torch.ones(seq_len, dtype=torch.float32, device=hidden_states.device)
            
            compressed_batch.append(compressed_sample)
            compressed_masks.append(new_mask)
        
        # Stack batch
        compressed_hidden_states = torch.stack(compressed_batch)
        compressed_attention_mask = torch.stack(compressed_masks) if attention_mask is not None else None
        
        # Calculate compression statistics
        compression_ratio = total_original_tokens / total_compressed_tokens if total_compressed_tokens > 0 else 1.0
        compression_stats = {
            'original_tokens': total_original_tokens,
            'compressed_tokens': total_compressed_tokens,
            'compression_ratio': compression_ratio,
        }
        
        return compressed_hidden_states, compressed_attention_mask, compression_stats

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for SpeechEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        kwargs_encoder = {
            argument[len("encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path,
                    **kwargs_encoder,
                    return_unused_kwargs=True,
                )

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path,
                    **kwargs_decoder,
                    return_unused_kwargs=True,
                )

                kwargs_decoder["config"] = decoder_config

            decoder = AutoModelForCausalLM.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder
            )

        # instantiate config with corresponding kwargs
        config = SpeechLMConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs
        )

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
        self,
        audio_input_features: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        audio_attention_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_compression_threshold: Optional[float] = None,
        weighted: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop(
                "num_items_in_batch", None
            )

        # we assume that if we are using cache then we are caching encoder_outputs
        if not use_cache or (use_cache and past_key_values is None):

            if self.config.add_pre_adapter:
                audio_input_features = self.pre_adapter(
                    audio_input_features, attention_mask=audio_attention_mask
                )
                audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                    audio_input_features.shape[1], audio_attention_mask
                )

            encoder_outputs = self.encoder(
                audio_input_features,
                attention_mask=audio_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            encoder_hidden_states = encoder_outputs[0]

            ##########################
            # (Optional) ADAPTER
            ##########################
            # we project the encoder outputs if we haven't done it
            # within the adapter
            if hasattr(self, "enc_to_dec_proj"):
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

            ##########################
            # TOKEN COMPRESSION (NEW)
            ##########################
            if token_compression_threshold is not None:
                if audio_attention_mask is not None:
                    compression_mask = self.encoder._get_feature_vector_attention_mask(
                        encoder_hidden_states.shape[1], audio_attention_mask
                    )
                else:
                    compression_mask = None
                
                encoder_hidden_states, encoder_outputs_mask, compression_stats = self.compress_tokens_by_similarity(
                    encoder_hidden_states,
                    attention_mask=compression_mask,
                    similarity_threshold=token_compression_threshold,
                    weighted=weighted
                )
                
                # Store compression stats for logging (accessible via model attributes)
                if not hasattr(self, 'compression_stats_history'):
                    self.compression_stats_history = []
                self.compression_stats_history.append(compression_stats)
                
                # Keep only last 100 stats to avoid memory issues
                if len(self.compression_stats_history) > 100:
                    self.compression_stats_history = self.compression_stats_history[-100:]
            else:
                # Original attention mask computation
                if audio_attention_mask is not None:
                    encoder_outputs_mask = self.encoder._get_feature_vector_attention_mask(
                        encoder_hidden_states.shape[1], audio_attention_mask
                    )
                else:
                    encoder_outputs_mask = torch.ones(
                        encoder_hidden_states.shape[:2],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

        # extract input embeds from the decoder
        decoder_input_embs = self.decoder.get_input_embeddings()(input_ids)

        # If we are not using the cache, or it's the first pass with the cache on.
        # Hence, we need to build new inputs for the decoder
        if not use_cache or (use_cache and past_key_values is None):
            # prepend audio representations to the text input embeddings
            decoder_input_embs = torch.cat(
                [encoder_hidden_states, decoder_input_embs], dim=1
            )

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [encoder_outputs_mask, attention_mask], dim=1
                )

        if logits_to_keep == 0:
            logits_to_keep = (
                labels.shape[1] if labels is not None else input_ids.shape[1]
            )

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_input_embs,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **kwargs_decoder,
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=decoder_outputs.logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                ignore_index=self.config.decoder.pad_token_id,
                **kwargs,
            )

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the SpeechEncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

    def generate(self, *args, **kwargs):
        if hasattr(self, "audio_attention_mask"):
            del self.audio_attention_mask
        return super().generate(*args, **kwargs)

    def can_generate(self):
        return True


__all__ = ["SpeechLMPreTrainedModel", "SpeechLMForConditionalGeneration"]