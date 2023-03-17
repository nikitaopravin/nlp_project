import streamlit as st
import torch
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import textwrap
import gdown
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.header('Цифровой собеседник')


path = '../data/gen_model.pt'
if not os.path.isfile(path):
    url = "https://drive.google.com/file/d/1zS7vuIJ7jgmvGgcS4CpzIs_wg8VdWqBI/view?usp=share_link"
    gdown.download(url=url, output=path, fuzzy=True)

status = st.radio('Личность модели', ('Regular', 'I want to believe'))
answer_len = st.slider('Разговорчивость', 0, 200, 50)
temp = st.slider('Креативность', 1., 10., 1.)

if status == 'Regular':

    tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    model = GPT2LMHeadModel.from_pretrained(
        'sberbank-ai/rugpt3small_based_on_gpt2',
        output_attentions = False,
        output_hidden_states = False,
    )

else:

    tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    model = GPT2LMHeadModel.from_pretrained(
        'sberbank-ai/rugpt3small_based_on_gpt2',
        output_attentions = False,
        output_hidden_states = False,
    )
    model.load_state_dict(torch.load(path))

prompt = st.text_input('Введите фразу', 'Привет')

model.to(device)
with torch.inference_mode():
    prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
    out = model.generate(
        input_ids=prompt,
        max_length=answer_len,
        num_beams=3,
        do_sample=True,
        temperature=temp,
        top_k=30,
        top_p=0.95 / (temp / 8),
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        ).cpu().numpy()

st.write(textwrap.fill(tokenizer.decode(out[0]), 100))

