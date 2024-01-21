import io
import os
from random import randint

import pandas as pd
import streamlit as st
import torch
from PIL import Image
import wandb
from transformers import BlipProcessor, BlipForConditionalGeneration
from wonderwords import RandomWord

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


def get_caption_for_image(image):
    inputs = processor(image, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(outputs[0], skip_special_tokens=True)


st.title("hellow!")
st.caption("эксперимент")
r_word = RandomWord()
exp_name = " ".join([r_word.word(include_parts_of_speech=["adjectives"], word_min_length=4, word_max_length=16),
                     r_word.word(include_parts_of_speech=["nouns"], word_min_length=4, word_max_length=16)])
st.subheader(exp_name)

exp_path = os.path.join("experiments", exp_name)
os.mkdir(exp_path)
output_dir = os.path.join("output", exp_name)
os.mkdir(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"работать будем на {device}")

prefix = st.text_input("Выберите префикс для подсказок")

result_df = None

uploaded_files = st.file_uploader("Выберите картинки для создания выборки", accept_multiple_files=True)
if prefix and uploaded_files:
    i = 0.0
    files_with_captions_list: list[dict] = []
    progress_bar = st.progress(value=0, text="Получаем подсказки для картинок...")

    for uploaded_file in uploaded_files:
        i += 1
        img = Image.open(io.BytesIO(uploaded_file.read()))
        img.save(os.path.join(exp_path, uploaded_file.name))
        files_with_captions_list.append({"file_name": uploaded_file.name,
                                         "text": f"{prefix} {get_caption_for_image(img)}"})
        progress_bar.progress(value=(i / len(uploaded_files)), text="Получаем подсказки для картинок...")

    result_df = pd.DataFrame.from_records(files_with_captions_list)
    st.dataframe(result_df)

if result_df is not None:
    result_df.to_json(os.path.join(exp_path, "metadata.jsonl"), orient='records', lines=True)
    wandb.login(key="a96e329746dd7aa553776a2572f48461710ee0e8")

    model_name = "runwayml/stable-diffusion-v1-5"
    run_command: list[str] = ["accelerate launch --mixed_precision=no --num_processes=1",
                              "train_text_to_image_lora.py",
                              f"--pretrained_model_name_or_path=\"{model_name}\"",
                              f"--train_data_dir=\"{exp_path}\"",
                              "--dataloader_num_workers=8",
                              "--resolution=512",
                              "--center_crop --random_flip",
                              "--train_batch_size=2",
                              "--gradient_accumulation_steps=4",
                              "--max_train_steps=1000",
                              "--num_train_epochs=10",
                              "--learning_rate=1e-04",
                              "--max_grad_norm=1",
                              "--lr_scheduler=cosine",
                              "--lr_warmup_steps=0",
                              f"--output_dir=\"{output_dir}\"",
                              "--report_to=wandb",
                              "--checkpointing_steps=100",
                              f"--validation_prompt=\"{result_df['text'].values[0]}\"",
                              f"--seed={randint(1,175465835)}"]
    st.code(" ".join(run_command), language="bash")
