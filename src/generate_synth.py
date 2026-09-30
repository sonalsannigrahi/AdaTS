from datasets import load_dataset
import tyro
import logging
from kokoro import KPipeline
import soundfile as sf
from tqdm import tqdm
import os


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def main(output_dir: str):
    data = load_dataset("CohereForAI/aya_dataset", split="train")

    eng_texts = [d["inputs"] for d in data if d["language_code"] == "eng"]
    logger.info(f"Number of English texts: {len(eng_texts)}")

    pipeline = KPipeline(lang_code="a")

    for outer_i, text in tqdm(enumerate(eng_texts), desc="item", total=len(eng_texts)):
        # print(text)
        generator = pipeline(
            text,
            voice="af_heart",  # <= change voice here
            speed=1,
            # split_pattern=r"\n+",
        )
        # for i, (gs, ps, audio) in enumerate(generator):
        # print(i)  # i => index
        # print(gs) # gs => graphemes/text
        # print(ps) # ps => phonemes
        # display(Audio(data=audio, rate=24000, autoplay=i==0))
        gs, ps, audio = next(generator)
        print()
        sf.write(
            os.path.join(output_dir, f"{outer_i}.wav"), audio, 24000
        )  # save each audio file


if __name__ == "__main__":
    tyro.cli(main)
