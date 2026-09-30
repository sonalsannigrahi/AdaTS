# This Gradio app allows the user to select a model, record audio, choose a task, and select a language.
# The model is loaded when the user clicks the "Load Model" button.
# The audio is processed based on the selected task and language, and the output is displayed in a textbox.

import gradio as gr
import os
import glob
from transformers import pipeline
from transformers.models.speechlm import (
    SpeechLMProcessor,
    SpeechLMForConditionalGeneration,
)
import torch
import numpy as np
import librosa


# Define the path to the models
MODELS_PATH = "/mnt/home/giuseppe/myscratch/speech_lm"

# List of available models
models = glob.glob(f"{MODELS_PATH}/**/checkpoint-*", recursive=True)


# Function to load the selected model
def load_model(model_name):
    model = SpeechLMForConditionalGeneration.from_pretrained(model_name)
    model.eval().to("cuda:0")
    try:
        processor = SpeechLMProcessor.from_pretrained(model_name)
    except Exception as e:
        print(e)
        processor = SpeechLMProcessor.from_encoder_decoder_pretrained(
            "facebook/w2v-bert-2.0", "utter-project/EuroLLM-1.7B"
        )
    return model, processor


# Function to process the audio based on the selected task and language
def process_audio(audio, task, language, model_state):
    model, processor = model_state
    sr, y = audio

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # resample from sr to 16000hz
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    if task == "transcribe":
        inputs = processor(
            audio=y,
            sampling_rate=16000,
            task=task,
            lang=language,
        )
    else:
        return "not implemented yet!"

    inputs = {k: torch.tensor(v).to(model.device) for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    # output_ids = model.greedy_decoding(inputs, max_tokens=128)
    transcript = processor.batch_decode(output_ids, skip_special_tokens=True)

    return transcript


def make_visible_model_loaded(o):
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as demo:
    # Model selection
    model_dropdown = gr.Dropdown(choices=models, label="Select a Model")
    load_button = gr.Button("Load Model")
    model_state = gr.State(None)  # To store the loaded model

    # Audio recording
    audio_input = gr.Audio(label="Record Audio", sources="microphone")

    # Task and language selection
    task_dropdown = gr.Dropdown(
        choices=["transcribe", "translate"], label="Select Task", visible=False
    )
    language_dropdown = gr.Dropdown(
        choices=["en", "it", "pt"], label="Select Language", visible=False
    )

    # Process button
    process_button = gr.Button("Process Audio", visible=False)

    # Output textbox
    output_textbox = gr.Textbox(label="Output", visible=False)

    # Load model when the "Load Model" button is clicked
    load_button.click(fn=load_model, inputs=model_dropdown, outputs=model_state).then(
        make_visible_model_loaded,
        outputs=[task_dropdown, language_dropdown, process_button, output_textbox],
    )

    # Process audio when the "Process Audio" button is clicked
    process_button.click(
        fn=process_audio,
        inputs=[audio_input, task_dropdown, language_dropdown, model_state],
        outputs=output_textbox,
    )

# Launch the interface
demo.launch(show_error=True)
