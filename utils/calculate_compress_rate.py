"""
Analyze compression rates on different datasets without running full model.

Usage:
    python analyze_compression_rate.py --threshold 0.6
    python analyze_compression_rate.py --threshold 0.6 --max_samples 100
    python analyze_compression_rate.py --thresholds 0.5 0.6 0.7
"""

import torch
import argparse
from datasets import load_dataset
from transformers import Wav2Vec2Model, AutoProcessor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def compress_tokens(hidden_states, threshold, max_group_size=10):
    """
    Compress tokens by similarity (same logic as in modeling_speechlm.py).
    
    Args:
        hidden_states: Tensor of shape (seq_len, hidden_dim)
        threshold: Similarity threshold
        max_group_size: Maximum tokens per group
        
    Returns:
        num_compressed: Number of tokens after compression
    """
    seq_len = hidden_states.shape[0]
    
    if seq_len <= 1:
        return seq_len
    
    # Compute similarities
    normalized = F.normalize(hidden_states, p=2, dim=-1)
    similarities = torch.sum(normalized[:-1] * normalized[1:], dim=-1)
    
    # Group tokens
    groups = []
    current_group = [0]
    
    for i, sim in enumerate(similarities):
        should_merge = sim >= threshold and len(current_group) < max_group_size
        
        if should_merge:
            current_group.append(i + 1)
        else:
            if len(current_group) > 0:
                groups.append(current_group)
            current_group = [i + 1]
    
    if len(current_group) > 0:
        groups.append(current_group)
    
    return len(groups)


def analyze_dataset(dataset_name, split, encoder, processor, threshold, max_samples=None):
    """Analyze compression rate for a dataset."""
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} - {split}")
    print(f"{'='*80}")
    
    # Load dataset
    if dataset_name == "librispeech_asr":
        dataset = load_dataset("librispeech_asr", split=split, trust_remote_code=True)
    elif dataset_name == "voxpopuli":
        dataset = load_dataset("facebook/voxpopuli", "en", split=split, trust_remote_code=True)
    elif dataset_name == "fleurs":
        dataset = load_dataset("google/fleurs", "en_us", split=split, trust_remote_code=True)
    elif dataset_name == "spoken":
        dataset = load_dataset("alinet/spoken_squad", "WER44", split = split,trust_remote_code=True)
    elif dataset_name == "slue":
        dataset = load_dataset("asapp/slue-phase-2", "sqa5", split = split, trust_remote_code=True)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Samples: {len(dataset)}")
    
    # Statistics
    total_original = 0
    total_compressed = 0
    compression_ratios = []
    audio_durations = []
    
    # Process samples
    for sample in tqdm(dataset, desc="Processing"):
        audio = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        
        # Resample if needed
        if sr != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(torch.from_numpy(audio)).numpy()
        
        # Process with encoder
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(encoder.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = encoder(**inputs)
            hidden_states = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
        
        original_tokens = hidden_states.shape[0]
        compressed_tokens = compress_tokens(hidden_states, threshold)
        
        total_original += original_tokens
        total_compressed += compressed_tokens
        compression_ratios.append(original_tokens / compressed_tokens)
        audio_durations.append(len(audio) / 16000)
    
    # Statistics
    avg_ratio = np.mean(compression_ratios)
    std_ratio = np.std(compression_ratios)
    total_audio = sum(audio_durations)
    
    print(f"\nResults:")
    print(f"  Total audio: {total_audio/60:.2f} minutes")
    print(f"  Total original tokens: {total_original:,}")
    print(f"  Total compressed tokens: {total_compressed:,}")
    print(f"  Overall compression ratio: {total_original/total_compressed:.2f}x")
    print(f"  Average compression ratio: {avg_ratio:.2f}x (±{std_ratio:.2f})")
    print(f"  Min compression: {min(compression_ratios):.2f}x")
    print(f"  Max compression: {max(compression_ratios):.2f}x")
    
    return {
        'dataset': dataset_name,
        'split': split,
        'samples': len(dataset),
        'total_audio_minutes': total_audio / 60,
        'total_original': total_original,
        'total_compressed': total_compressed,
        'overall_ratio': total_original / total_compressed,
        'avg_ratio': avg_ratio,
        'std_ratio': std_ratio,
        'min_ratio': min(compression_ratios),
        'max_ratio': max(compression_ratios),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="facebook/w2v-bert-2.0",
                       help="Encoder model")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Single threshold to test")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                       help="Multiple thresholds to test")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples per dataset (for testing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Determine thresholds to test
    if args.thresholds:
        thresholds = args.thresholds
    elif args.threshold:
        thresholds = [args.threshold]
    else:
        thresholds = [0.5, 0.6, 0.7]
    
    # Load encoder once
    print("Loading encoder...")
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    encoder.eval()
    encoder = encoder.to(args.device)
    print(f"Encoder loaded on {args.device}")
    
    # Datasets to analyze
    datasets_config = [
        ("librispeech_asr", "test.clean"),
        ("librispeech_asr", "test.other"),
        ("voxpopuli", "test"),
        ("fleurs", "test"),
        # ("spoken", "test"),
        # ("slue","test")
    ]
    
    # Run analysis
    all_results = []
    
    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'='*80}")
        
        threshold_results = []
        
        for dataset_name, split in datasets_config:
            try:
                result = analyze_dataset(
                    dataset_name,
                    split,
                    encoder,
                    processor,
                    threshold,
                    args.max_samples
                )
                result['threshold'] = threshold
                threshold_results.append(result)
            except Exception as e:
                print(f"Error processing {dataset_name}/{split}: {e}")
        
        all_results.extend(threshold_results)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Dataset':<20} {'Split':<15} {'Threshold':<10} {'Samples':<8} {'Audio(min)':<12} {'Ratio':<8}")
    print("-"*80)
    
    for result in all_results:
        print(
            f"{result['dataset']:<20} "
            f"{result['split']:<15} "
            f"{result['threshold']:<10.2f} "
            f"{result['samples']:<8} "
            f"{result['total_audio_minutes']:<12.2f} "
            f"{result['overall_ratio']:<8.2f}x"
        )
    
    # Summary by threshold
    print("\n" + "="*80)
    print("AVERAGE COMPRESSION BY THRESHOLD")
    print("="*80)
    
    for threshold in thresholds:
        threshold_results = [r for r in all_results if r['threshold'] == threshold]
        avg_ratio = np.mean([r['overall_ratio'] for r in threshold_results])
        print(f"Threshold {threshold:.2f}: {avg_ratio:.2f}x compression (avg across datasets)")


if __name__ == "__main__":
    main()