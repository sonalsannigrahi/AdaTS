# AdaTS: Adaptive Token Sampling for Efficient Speech Language Models

Implementation for the paper: AdaTS: Adaptive Token Sampling for Efficient Speech Language Models presented at INTERSPEECH 2026 and Workshop on Multimodal Intelligence at ICLR 2026.

Abstract:

Speech Language Models (SLM) have demonstrated strong capabilities in end-to-end speech understanding and reasoning tasks by incorporating speech tokens into a Large Language Model (LLM). However, most common designs are i) token-intensive, since a large part of the LLM context is allotted to audio tokens, and ii) inefficient, as audio representation is often redundant, hindering SLMs' capabilities to handle long-form tasks. To address token inefficiency, we propose a dynamic sampling method that adaptively groups and merges speech tokens where the signal is less information-dense. Our approach reduces speech length by 2x on average while yielding performance comparable to or better than standard convolutional downsampling across Speech Recognition (ASR), Speech Question-Answering (SQA), and Speech Translation (ST). Through extensive empirical analysis, we demonstrate the effectiveness of this strategy in preserving speech content and exhibiting general speech understanding capabilities, while substantially reducing token redundancy and inference cost by 40%. We release all of our code to the community.


Code:

We implement the modelling code as part of transformers, so installation must be done from source and adding the speechlm modelling, preprocessing, and configuration code in transformers/src/transformers/models directory. 

Pip install remaining packages from requirements.txt.

For training and evaluation data setup and follow original repositories for download instructions. We provide utils to load these datasets, adding new datasets for training is also straightforward by following the relevant utils script.

To run the training, use the train.py script providing configs from the config directory. 

