"""
Minimal FLOPs calculator for SpeechLM.

Usage:
    python simple_flops.py ./model
"""

import sys
import torch
from calflops import calculate_flops
from transformers.models.speechlm import SpeechLMProcessor, SpeechLMForConditionalGeneration


def compute_flops(model_path):
    # Load
    processor = SpeechLMProcessor.from_pretrained(model_path)
    model = SpeechLMForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    
    # Create inputs (5 seconds audio, 50 tokens text)
    audio = torch.randn(1, 2*80000)  # 5s at 16kHz
    inputs = processor(
        audio=[audio.numpy()],
        text=["test " * 50],
        sampling_rate=16000,
        return_tensors="pt",
        task="transcribe",
        target_lang="en",
    )
    
    # Calculate
    flops, macs, params = calculate_flops(
        model=model,
        kwargs=inputs,
        print_results=True,
    )
    
    print(flops)
    # Summary
    print(f"\nSummary:")
    print(f"  Total FLOPs: {float(flops)/1e12:.2f} TFLOPs")
    print(f"  Parameters: {params/1e9:.2f}B")
    print(f"  FLOPs per audio second: {flops/5/1e9:.2f} GFLOPs/s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_flops.py <model_path>")
        sys.exit(1)
    
    compute_flops(sys.argv[1])