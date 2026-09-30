from datasets import load_dataset
from transformers.models.speechlm import (
    SpeechLMProcessor,
    SpeechLMForConditionalGeneration,
)
from transformers import AutoTokenizer, AutoConfig
import tyro


def main(audio_encoder: str, text_decoder: str):

    load_dataset("pykeio/librivox-tracks")
    processor = SpeechLMProcessor.from_encoder_decoder_pretrained(
        audio_encoder, text_decoder, add_eos_token=True
    )

    model = SpeechLMForConditionalGeneration.from_encoder_decoder_pretrained(
        audio_encoder,
        text_decoder,
    )
    AutoTokenizer.from_pretrained(text_decoder)
    AutoConfig.from_pretrained(text_decoder)
    print("DONE", audio_encoder, text_decoder)


if __name__ == "__main__":
    tyro.cli(main)
