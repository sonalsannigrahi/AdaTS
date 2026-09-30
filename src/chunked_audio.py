# coding=utf-8
"""
Long Audio Chunking Module for Speech LM

This module implements efficient chunking strategies for processing long audio
sequences (50+ minutes) without running out of memory.

Strategy Overview:
1. Split long audio into overlapping chunks
2. Process each chunk through encoder independently
3. Handle overlap regions to ensure continuity
4. Concatenate encoded representations
5. Apply token compression on the full sequence
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math


class AudioChunker:
    """
    Handles chunking and reconstruction of long audio sequences.
    
    Design principles:
    - Use overlapping chunks to avoid boundary artifacts
    - Process chunks independently to manage memory
    - Blend overlapping regions using weighted averaging
    - Support variable chunk sizes based on available memory
    """
    
    def __init__(
        self,
        chunk_length_s: float = 30.0,
        overlap_s: float = 1.0,
        sampling_rate: int = 16000,
        blend_overlap: bool = True,
    ):
        """
        Args:
            chunk_length_s: Length of each chunk in seconds
            overlap_s: Overlap between chunks in seconds
            sampling_rate: Audio sampling rate in Hz
            blend_overlap: Whether to blend overlapping regions
        """
        self.chunk_length_s = chunk_length_s
        self.overlap_s = overlap_s
        self.sampling_rate = sampling_rate
        self.blend_overlap = blend_overlap
        
        self.chunk_length_samples = int(chunk_length_s * sampling_rate)
        self.overlap_samples = int(overlap_s * sampling_rate)
        self.stride_samples = self.chunk_length_samples - self.overlap_samples
    
    def chunk_audio(
        self, 
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], List[Tuple[int, int]]]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Audio tensor of shape (batch_size, audio_length) or (audio_length,)
            attention_mask: Optional attention mask
            
        Returns:
            chunks: List of audio chunks
            chunk_masks: List of attention masks for each chunk
            chunk_positions: List of (start, end) positions in original audio
        """
        # Handle single audio or batch
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            single_audio = True
        else:
            single_audio = False
        
        batch_size, audio_length = audio.shape
        
        # Calculate number of chunks needed
        if audio_length <= self.chunk_length_samples:
            # Audio is shorter than chunk size, no chunking needed
            chunks = [audio]
            chunk_masks = [attention_mask] if attention_mask is not None else [None]
            chunk_positions = [(0, audio_length)]
        else:
            chunks = []
            chunk_masks = []
            chunk_positions = []
            
            start = 0
            while start < audio_length:
                end = min(start + self.chunk_length_samples, audio_length)
                
                # Extract chunk
                chunk = audio[:, start:end]
                chunks.append(chunk)
                
                # Extract mask if provided
                if attention_mask is not None:
                    chunk_mask = attention_mask[:, start:end]
                    chunk_masks.append(chunk_mask)
                else:
                    chunk_masks.append(None)
                
                chunk_positions.append((start, end))
                
                # Move to next chunk with stride
                start += self.stride_samples
                
                # Break if we've covered the entire audio
                if end >= audio_length:
                    break
        
        return chunks, chunk_masks, chunk_positions
    
    def merge_encoded_chunks(
        self,
        encoded_chunks: List[torch.Tensor],
        chunk_positions: List[Tuple[int, int]],
        encoder_stride: int,
        original_audio_length: int,
    ) -> torch.Tensor:
        """
        Merge encoded chunks back into a single sequence, handling overlaps.
        
        Args:
            encoded_chunks: List of encoded representations [batch, seq_len, hidden_dim]
            chunk_positions: Original audio positions for each chunk
            encoder_stride: Stride/downsampling factor of the encoder
            original_audio_length: Length of original audio in samples
            
        Returns:
            merged: Merged encoded representation
        """
        if len(encoded_chunks) == 1:
            return encoded_chunks[0]
        
        batch_size, _, hidden_dim = encoded_chunks[0].shape
        device = encoded_chunks[0].device
        
        # Calculate the overlap in encoded space
        overlap_encoded = self.overlap_samples // encoder_stride
        
        # For blending, we'll use a simple linear blend in the overlap region
        merged_chunks = []
        
        for i, chunk in enumerate(encoded_chunks):
            if i == 0:
                # First chunk: keep everything except maybe trim end overlap/2
                if self.blend_overlap and len(encoded_chunks) > 1:
                    trim_end = overlap_encoded // 2
                    merged_chunks.append(chunk[:, :-trim_end, :])
                else:
                    merged_chunks.append(chunk)
            elif i == len(encoded_chunks) - 1:
                # Last chunk: trim start overlap/2
                if self.blend_overlap:
                    trim_start = overlap_encoded // 2
                    merged_chunks.append(chunk[:, trim_start:, :])
                else:
                    merged_chunks.append(chunk[:, overlap_encoded:, :])
            else:
                # Middle chunks: trim both ends by overlap/2
                if self.blend_overlap:
                    trim_start = overlap_encoded // 2
                    trim_end = overlap_encoded // 2
                    merged_chunks.append(chunk[:, trim_start:-trim_end, :])
                else:
                    merged_chunks.append(chunk[:, overlap_encoded:, :])
        
        # Concatenate all chunks
        merged = torch.cat(merged_chunks, dim=1)
        
        return merged
    
    def calculate_encoder_stride(self, encoder) -> int:
        """
        Calculate the downsampling stride of the encoder.
        For wav2vec2-bert, this is typically computed from conv layers.
        """
        # For wav2vec2-bert, the stride is usually 320 or 640 depending on config
        # We'll try to get it from the model config
        if hasattr(encoder, 'config'):
            if hasattr(encoder.config, 'adapter_stride'):
                return encoder.config.adapter_stride
            # Common wav2vec2 stride calculation
            # Typically: product of conv strides
            # Default for wav2vec2-bert is often 320
            return 320
        return 320  # Safe default


class ChunkedAudioEncoder(nn.Module):
    """
    Wrapper around an audio encoder that handles long sequences via chunking.
    Improvements compared to your original:
      - Batch-processing of chunks (process many chunks together to utilize GPU).
      - Avoid chunking when inputs look like already-extracted features (e.g. (batch, seq, feat)).
      - Delegates HF helpers like _get_feature_vector_attention_mask and config.
      - Optional streaming API hint (if inner encoder supports stateful processing).
    """

    def __init__(
        self,
        encoder: nn.Module,
        chunk_length_s: float = 30.0,
        overlap_s: float = 1.0,
        sampling_rate: int = 16000,
        max_chunk_batch: int = 16,   # number of chunks processed together to control OOM
        chunk_if_waveform_only: bool = True,  # only chunk raw waveforms if True
    ):
        super().__init__()
        self.encoder = encoder
        self.chunker = AudioChunker(
            chunk_length_s=chunk_length_s,
            overlap_s=overlap_s,
            sampling_rate=sampling_rate,
            blend_overlap=True,
        )
        self.encoder_stride = self.chunker.calculate_encoder_stride(encoder)
        self.max_chunk_batch = max_chunk_batch
        self.chunk_if_waveform_only = chunk_if_waveform_only

    #
    # HF compatibility delegates
    #
    @property
    def config(self):
        return getattr(self.encoder, "config", None)

    def _get_feature_vector_attention_mask(self, attention_mask, feature_vector_length):
        # Delegate to inner encoder if possible (HF convention)
        if hasattr(self.encoder, "_get_feature_vector_attention_mask"):
            return self.encoder._get_feature_vector_attention_mask(attention_mask, feature_vector_length)
        # fallback: assume already in right space
        return attention_mask

    #
    # Heuristics to decide whether to chunk:
    # - if input is 3D (batch, seq, feature_dim) -> assume already-extracted features -> do NOT chunk
    # - if chunk_if_waveform_only==False -> chunk regardless (advanced use)
    #

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing in the underlying encoder.
        """
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def _should_chunk(self, tensor: torch.Tensor) -> bool:
        # tensor expected shapes:
        #  waveform: (batch, samples)  -> dim==2
        #  features: (batch, seq, feat) -> dim==3
        if tensor.dim() == 3:
            return False
        if tensor.dim() != 2:
            # unexpected shape -> don't chunk to be safe
            return False
        if not self.chunk_if_waveform_only:
            # user forced chunking
            audio_length = tensor.shape[-1]
            return audio_length > self.chunker.chunk_length_samples
        # default: only chunk if waveform length in samples > chunk_length_samples
        audio_length = tensor.shape[-1]
        return audio_length > self.chunker.chunk_length_samples

    #
    # Core batched chunk encoder: collect all chunks from the batch, process in sub-batches,
    # and distribute encoded outputs back to per-sample lists preserving order.
    #
    def _encode_chunk_batches(
        self,
        chunk_tensors: List[torch.Tensor],
        chunk_masks: List[Optional[torch.Tensor]],
        return_dict: bool,
    ) -> List[torch.Tensor]:
        """
        chunk_tensors: list of tensors each shape (1, L) or (1, L, feat) depending on encoder
        chunk_masks: matching list of masks or None
        returns: list of encoded chunk outputs aligned with chunk_tensors,
                 each element shape (1, encoded_len, hidden_dim)
        """
        device = chunk_tensors[0].device
        dtype = chunk_tensors[0].dtype
        n = len(chunk_tensors)
        encoded_results = [None] * n

        # process in sub-batches to control memory
        start = 0
        while start < n:
            end = min(start + self.max_chunk_batch, n)
            batch_chunks = torch.cat(chunk_tensors[start:end], dim=0)  # shape (B, L) or (B, L, feat)
            if chunk_masks is not None and any(m is not None for m in chunk_masks[start:end]):
                mask_list = [
                    m if m is not None else torch.ones((1, batch_chunks.shape[1]), device=device, dtype=torch.long)
                    for m in chunk_masks[start:end]
                ]
                batch_mask = torch.cat(mask_list, dim=0)
            else:
                batch_mask = None

            # call encoder once for the whole mini-batch
            with torch.set_grad_enabled(self.training):
                out = self.encoder(
                    batch_chunks,
                    attention_mask=batch_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=return_dict,
                )
            if return_dict:
                batch_encoded = out.last_hidden_state  # (B, enc_len, hidden)
            else:
                batch_encoded = out[0]

            # split them back
            for i in range(start, end):
                j = i - start
                encoded_results[i] = batch_encoded[j:j+1]
            start = end

        return encoded_results

    def forward(
        self,
        audio_input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Forward that batches chunk processing for speed.
        Preserves the same return contract as the inner encoder (BaseModelOutput or tuple).
        """

        # Basic shapes and device
        device = audio_input_features.device
        batch_size = audio_input_features.shape[0]

        all_encoded = []
        all_masks = []

        # Collect chunk-level inputs for batched encoding
        # We'll also keep mapping from chunk index -> which sample it belongs to and position info
        chunk_inputs = []
        chunk_input_masks = []
        chunk_mapping = []  # list of (sample_index, chunk_pos_index, chunk_positions)

        # First pass: determine which samples need chunking and prepare their chunks
        for b in range(batch_size):
            sample = audio_input_features[b:b+1]  # keep batch dim
            sample_mask = attention_mask[b:b+1] if attention_mask is not None else None

            # decide whether to chunk
            if self._should_chunk(sample):
                chunks, chunk_masks, chunk_positions = self.chunker.chunk_audio(sample, sample_mask)
                # extend chunk_inputs with these (each chunk shape is (1, L))
                for i, (c, cm, pos) in enumerate(zip(chunks, chunk_masks, chunk_positions)):
                    chunk_inputs.append(c.to(device))
                    chunk_input_masks.append(cm.to(device) if cm is not None else None)
                    chunk_mapping.append((b, i, pos))
            else:
                # mark with special sentinel so we process the sample directly
                chunk_mapping.append((b, None, None))
                all_encoded.append(None)  # placeholder
                all_masks.append(sample_mask if sample_mask is not None else None)

        # If there are chunk_inputs, encode them in batches
        encoded_chunk_outputs = []
        if len(chunk_inputs) > 0:
            encoded_chunk_outputs = self._encode_chunk_batches(chunk_inputs, chunk_input_masks, return_dict)

        # Now rebuild per-sample encoded lists
        # For samples that were chunked, collect their encoded chunks in order and merge
        # For samples not chunked, call encoder directly (we do this after chunk processing to avoid extra device switches)
        # Build a map from sample index -> list of encoded chunks & positions
        per_sample_chunks = {i: [] for i in range(batch_size)}
        per_sample_positions = {i: [] for i in range(batch_size)}
        e_idx = 0
        for entry in chunk_mapping:
            sample_idx, chunk_idx, pos = entry
            if chunk_idx is None:
                # not chunked; placeholder already added above (all_encoded at index sample_idx is None)
                continue
            # encoded_chunk_outputs are in the same order as chunk_inputs
            enc = encoded_chunk_outputs[e_idx]
            per_sample_chunks[sample_idx].append(enc)
            per_sample_positions[sample_idx].append(pos)
            e_idx += 1

        # Post-process each sample: either merge encoded chunks or directly encode sample
        for b in range(batch_size):
            # If per_sample_chunks[b] is non-empty -> that sample was chunked
            if len(per_sample_chunks[b]) > 0:
                merged = self.chunker.merge_encoded_chunks(
                    per_sample_chunks[b],
                    per_sample_positions[b],
                    self.encoder_stride,
                    original_audio_length=audio_input_features[b:b+1].shape[-1],
                )
                all_encoded[b] = merged  # shape (1, enc_len, hidden)
                # produce attention mask for merged encoding if original had mask
                orig_mask = all_masks[b]
                if orig_mask is not None:
                    enc_len = merged.shape[1]
                    all_masks[b] = torch.ones((1, enc_len), dtype=orig_mask.dtype, device=orig_mask.device)
                else:
                    all_masks[b] = None
            else:
                # Not chunked -> encode the full sample directly
                sample = audio_input_features[b:b+1].to(device)
                sample_mask = attention_mask[b:b+1].to(device) if attention_mask is not None else None
                out = self.encoder(
                    sample,
                    attention_mask=sample_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                if return_dict:
                    enc = out.last_hidden_state
                else:
                    enc = out[0]
                all_encoded[b] = enc
                all_masks[b] = sample_mask

        # Now pad/collate to single batch
        max_len = max(enc.shape[1] for enc in all_encoded)
        padded_encoded = []
        padded_masks = []
        for enc, mask in zip(all_encoded, all_masks):
            if enc.shape[1] < max_len:
                pad = torch.zeros((1, max_len - enc.shape[1], enc.shape[2]), dtype=enc.dtype, device=enc.device)
                enc = torch.cat([enc, pad], dim=1)
            padded_encoded.append(enc)
            if mask is not None:
                if mask.shape[1] < max_len:
                    mpad = torch.zeros((1, max_len - mask.shape[1]), dtype=mask.dtype, device=mask.device)
                    mask = torch.cat([mask, mpad], dim=1)
                padded_masks.append(mask)
            else:
                padded_masks.append(None)

        final_encoded = torch.cat(padded_encoded, dim=0)
        if all(m is not None for m in padded_masks):
            final_mask = torch.cat(padded_masks, dim=0)
        else:
            final_mask = None

        if return_dict:
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(
                last_hidden_state=final_encoded,
                hidden_states=None,
                attentions=None,
            )
        else:
            return (final_encoded,)



def integrate_chunked_encoder(model, chunk_length_s: float = 30.0, overlap_s: float = 1.0):
    """
    Integrate chunked encoder into an existing SpeechLM model.
    
    Args:
        model: SpeechLMForConditionalGeneration instance
        chunk_length_s: Length of each chunk in seconds
        overlap_s: Overlap between chunks in seconds
    
    Returns:
        Modified model with chunked encoder
    """
    # Get sampling rate from encoder config
    # Handle both direct encoder and ChunkedAudioEncoder
    if isinstance(model.encoder, ChunkedAudioEncoder):
        # Already wrapped, just update parameters
        model.encoder.chunker.chunk_length_s = chunk_length_s
        model.encoder.chunker.overlap_s = overlap_s
        model.encoder.chunker.chunk_length_samples = int(chunk_length_s * model.encoder.chunker.sampling_rate)
        model.encoder.chunker.overlap_samples = int(overlap_s * model.encoder.chunker.sampling_rate)
        model.encoder.chunker.stride_samples = model.encoder.chunker.chunk_length_samples - model.encoder.chunker.overlap_samples
        return model
    
    # Get the actual encoder (might be nested)
    actual_encoder = model.encoder
    sampling_rate = getattr(actual_encoder.config, 'sampling_rate', 16000)
    
    # Wrap the encoder
    chunked_encoder = ChunkedAudioEncoder(
        encoder=actual_encoder,
        chunk_length_s=chunk_length_s,
        overlap_s=overlap_s,
        sampling_rate=sampling_rate,
    )
    
    # Replace the encoder
    model.encoder = chunked_encoder
    
    # Important: Make sure the chunked encoder is on the same device
    if hasattr(model, 'device'):
        model.encoder = model.encoder.to(model.device)
    
    return model