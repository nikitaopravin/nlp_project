import streamlit as st
import torch
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import textwrap
import gdown
import os


stock_model_path = 'data/stock_model/'
if not os.path.isdir(stock_model_path):
    url = "https://drive.google.com/drive/folders/1U4rp3CwKLBY9w3AN7KvBlo2cyjE_XW0t?usp=share_link"
    gdown.download_folder(url=url, output=stock_model_path)

trained_model_path = 'data/trained_model/'
if not os.path.isdir(trained_model_path):
    url = "https://drive.google.com/drive/folders/1-bLrYaO9XNOJ1q_w4rFGwvdzr4qJjl2b?usp=share_link"
    gdown.download_folder(url=url, output=trained_model_path)

# tokenizer_path = '../data/tokenizer/'
# if not os.path.isdir(tokenizer_path):
#     url = "https://drive.google.com/drive/folders/1-hvEuHJ_K9BsbneYKLoG8bOivRdjJ2To?usp=share_link"
#     gdown.download_folder(url=url, output=tokenizer_path)

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

st.header('Цифровой собеседник')

status = st.radio('Личность модели', ('😀 Regular ', '👽 I want to believe'))
answer_len = st.slider('Разговорчивость (кол-во слов)', 0, 100, 50)
temp = float(st.slider('Креативность', 1, 5, 2))

if status == '😀 Regular':

    model = GPT2LMHeadModel.from_pretrained(
        stock_model_path,
        output_attentions = False,
        output_hidden_states = False,
    )

else:

    model = GPT2LMHeadModel.from_pretrained(
        trained_model_path,
        output_attentions = False,
        output_hidden_states = False,
    )

prompt = st.text_input('Введите фразу', 'Привет')

with torch.inference_mode():
    prompt = tokenizer.encode(prompt, return_tensors='pt')
    out = model.generate(
        input_ids=prompt,
        max_length=answer_len,
        num_beams=5,
        do_sample=True,
        temperature=temp,
        top_k=30,
        top_p=0.19 * temp,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        ).cpu().numpy()

st.write(textwrap.fill(tokenizer.decode(out[0]), 100))

