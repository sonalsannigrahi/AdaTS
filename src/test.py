from datasets import load_dataset
from transformers.models.speechlm import (
    SpeechLMProcessor,
    SpeechLMForConditionalGeneration,
)
import torch
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    # path = "/mnt/home/giuseppe/myscratch/speech_lm/w2v-eurollm1.7-fleurs"
    path = "/mnt/home/giuseppe/myscratch/speech_lm/w2v-eurollm1.7-librispeech_w-other"
    path = "/mnt/home/giuseppe/myscratch/speech_lm/w2v-eurollm1.7_mls-ps_h100/checkpoint-11500"

    model = SpeechLMForConditionalGeneration.from_pretrained(path)
    try:
        processor = SpeechLMProcessor.from_pretrained(path)
    except Exception as e:
        print(e)
        processor = SpeechLMProcessor.from_encoder_decoder_pretrained(
            "facebook/w2v-bert-2.0", "utter-project/EuroLLM-1.7B"
        )

    test_dataset = load_dataset("google/fleurs", "en_us", split="test")
    # test_dataset = load_dataset(
    #     "mozilla-foundation/common_voice_17_0",
    #     "en",
    #     split="test",
    #     streaming=True,
    #     trust_remote_code=True,
    # )

    model.eval().to("cuda:0")

    # with torch.no_grad():
    for data in tqdm(test_dataset, desc="Transcribing"):
        inputs = processor(
            audio=data["audio"]["array"],
            sampling_rate=16000,
            task="transcribe",
            lang="en",
        )
        inputs = {k: torch.tensor(v).to("cuda:0") for k, v in inputs.items()}
        output_ids = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        # output_ids = model.greedy_decoding(inputs, max_tokens=128)
        transcript = processor.batch_decode(output_ids, skip_special_tokens=True)
        print(transcript)

        # for _ in range(30):
        #     output = model(**inputs)
        #     pred_ids = output.logits.argmax(-1)
        #     text = processor.batch_decode(pred_ids)
        #     print(text)

        #     # append pred_ids[0, -1] to inputs["text_input_ids"]
        #     inputs["text_input_ids"] = torch.cat(
        #         [inputs["text_input_ids"], pred_ids[:, -1].unsqueeze(0)], dim=1
        #     )
        #     # add a 1 to the attention mask
        #     inputs["text_attention_mask"] = torch.cat(
        #         [inputs["text_attention_mask"], torch.tensor([[1]]).to("cuda:0")],
        #         dim=1,
        #     )
