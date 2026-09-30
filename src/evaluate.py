import tyro
import time
import json
import glob
import yaml
import logging
from transformers.models.speechlm import (
    SpeechLMProcessor,
    SpeechLMForConditionalGeneration,
)
from tqdm import tqdm
import torch
from evaluation.normalizers import BasicTextNormalizer, EnglishTextNormalizer
import jiwer
from data_utils.utils import get_dataset
from training_utils import AudioTextDataCollator
import os
import pdb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_transcripts(model, processor, dataset, target_lang: str, batch_size):

    # wrap the dataset in a sequential dataloader
    data_collator = AudioTextDataCollator(
        processor, add_text=False, return_labels=False
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator, pin_memory=True
    )

    # with torch.no_grad():
    transcripts = list()

    for batch in tqdm(loader, desc="Transcribing", total=len(dataset) // batch_size):
        # for data in tqdm(dataset, desc="Transcribing", total=len(dataset)):
        batch = {
            k: v.to(
                device=model.device,
                dtype=model.dtype if k == "audio_input_features" else v.dtype,
            )
            for k, v in batch.items()
        }
        # if batch_size == 1:
        # batch.pop("attention_mask")
        # batch.pop("audio_attention_mask")

        output_ids = model.generate(
            **batch,
            pad_token_id=processor.tokenizer.pad_token_id,
            max_new_tokens=256,
            use_cache=True,
        )

        transcript = processor.batch_decode(output_ids, skip_special_tokens=True)
        transcripts.extend(transcript)

    return transcripts


def get_model_processor(model_name_or_path: str):
    model = SpeechLMForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.eval().to("cuda:0")
    try:
        processor = SpeechLMProcessor.from_pretrained(model_name_or_path)
    except:
        logger.warning(
            "Failed to load the processor. Trying to load it from the parent directory"
        )
        processor = SpeechLMProcessor.from_pretrained(
            os.path.dirname(model_name_or_path)
        )

    processor.tokenizer.padding_side = "left"
    return model, processor


def compute_metrics(
    references: list[str],
    transcripts: list[str],
    lang: str,
    whisper_normalize_text: bool = False,
):
    # we strip away any parenthesis and square brackets
    # references = [re.sub(r"[\[\]()]", "", r) for r in references]
    # transcriptions = [re.sub(r"[\[\]()]", "", r) for r in transcriptions]
    tr = list()
    for t in transcripts:
        tok = t.split(" ")
        try:
            l = int(tok[0])
            tr.append(" ".join(tok[1:]))
        except:
            tr.append(t)
    transcripts = tr

    if whisper_normalize_text:
        if lang == "en":
            normalizer = EnglishTextNormalizer()
        else:
            normalizer = BasicTextNormalizer()

        references = [normalizer(r).strip() for r in references]
        transcripts = [normalizer(t).strip() for t in transcripts]

        # if the reference is empty after the normalization (it might happen if the whole text is in between parentheses), we set it to the transcription so that wer/cer are 0.0
        references = [t if r == "" else r for r, t in zip(references, transcripts)]

    return {
        "wer": jiwer.wer(references, transcripts),
        "cer": jiwer.cer(references, transcripts),
    }


def main(
    task_name: str,
    model_name_or_path: str,
    output_file: str,
    nproc: int = 4,
    dry_run: bool = False,
    batch_size: int = 1,
):
    # 1. load the configuration file
    eval_configfile = "./config/eval_datasets.yaml"
    with open(eval_configfile) as file:
        config = yaml.safe_load(file)

    task_config = config[task_name]
    dataset_name = task_config[
        "dataset_name"
    ]  # name used to refer to the dataset internallay (e.g., for the loading functions)
    logger.info(f"Task config: {task_config}, dataset name: {dataset_name}")

    # Load all possible configs in this task.
    dataset_configs = task_config["configs"]
    if not isinstance(dataset_configs, list):
        dataset_configs = [dataset_configs]
    logger.info(f"Loaded task configs: {dataset_configs}")

    # Initialize the output dictionary
    out_dict = task_config
    out_dict["model_name_or_path"] = model_name_or_path
    transcription_time = 0

    logger.info(f"Loading model {model_name_or_path}")
    model, processor = get_model_processor(model_name_or_path)

    splits = [
        dc.get("split") if not isinstance(dc, str) else "test" for dc in dataset_configs
    ]
    logger.info(f"Loading dataset with splits {splits}")
    # data = load_from_disk(os.path.join(os.getenv("LOCAL_DATASETS_DIR"), dataset_name))
    data = get_dataset(dataset_name, splits=splits, n_proc=nproc)

    for dc in dataset_configs:
        logger.info(f"Evaluating config {dc}")

        if isinstance(dc, str):
            cname = lang = dc
            split = "test"  # some sensible default
        else:
            cname = dc["name"]
            lang = dc["lang"]
            split = dc["split"]

        test_data = data[split]
        test_data = test_data.take(100) if dry_run else test_data  # subset for testing

        stime = time.time()
        transcripts = get_transcripts(model, processor, test_data, lang, batch_size)
        logger.info("#########")
        logger.info("Some transcripts")
        logger.info(transcripts[:3])
        tr_time = time.time() - stime
        transcription_time += tr_time
        logger.info("Transcription time: %f", tr_time)

        metrics = compute_metrics(
            test_data["text"],  ## every dataset in this codebase has "text"
            transcripts,
            lang,
            task_config["whisper_normalize_text"],
        )
        metrics["transcription_time"] = tr_time
        metrics["transcripts"] = transcripts
        if dry_run:
            metrics["references"] = test_data["text"]
        out_dict[cname] = metrics

    out_dict["transcription_time"] = transcription_time
    with open(output_file, "w") as file:
        json.dump(out_dict, file, indent=2)


if __name__ == "__main__":
    tyro.cli(main)
