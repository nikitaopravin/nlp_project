import streamlit as st
import requests
import pickle
import pandas as pd
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
#from rnn_preprocessing import padding, data_preprocessing, preprocess_single_string

import re
import string
from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))

st.set_page_config(layout='wide')

def data_preprocessing(text: str) -> str:
    """preprocessing string: lowercase, removing html-tags, punctuation and stopwords

    Args:
        text (str): input string for preprocessing

    Returns:
        str: preprocessed string
    """    

    text = text.lower()
    text = re.sub('<.*?>', '', text) # html tags
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    #text = [word for word in text.split() if word not in stop_words] 
    text = ' '.join(text)
    return text

def get_words_by_freq(sorted_words: list, n: int = 10) -> list:
    return list(filter(lambda x: x[1] > n, sorted_words))

def padding(review_int: list, seq_len: int) -> np.array:
    """Make left-sided padding for input list of tokens

    Args:
        review_int (list): input list of tokens
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros

    Returns:
        np.array: padded sequences
    """    
    features = np.zeros((len(review_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)
            
    return features

def preprocess_single_string(
    input_string: str, 
    seq_len: int, 
    vocab_to_int: dict,
    ) -> torch.tensor:
    """Function for all preprocessing steps on a single string

    Args:
        input_string (str): input single string for preprocessing
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros
        vocab_to_int (dict, optional): word corpus {'word' : int index}. Defaults to vocab_to_int.

    Returns:
        list: preprocessed string
    """    

    preprocessed_string = data_preprocessing(input_string)
    result_list = []
    for word in preprocessed_string.split():
        try: 
            result_list.append(vocab_to_int[word])
        except KeyError as e:
            print(f'{e}: not in dictionary!')
    result_padded = padding([result_list], seq_len)[0]

    return torch.tensor(result_padded)



device = 'cpu'

VOCAB_SIZE = 2457
EMBEDDING_DIM = 128
HIDDEN_DIM = 32
N_LAYERS = 2

class sentimentLSTM(nn.Module):
    """
    The LSTM model that will be used to perform Sentiment analysis.
    """
    
    def __init__(self, 
    
                vocab_size: int, 
                embedding_dim: int, 
                hidden_dim: int,
                n_layers: int,                
                drop_prob=0.5) -> None:
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            n_layers, 
                            #dropout=drop_prob, 
                            batch_first=True)
        
        #self.dropout = nn.Dropout()
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        self.batch_size = x.size(0)
        hidden = self.init_hidden()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.fc(lstm_out)
        #out = self.dropout(out)

        sig_out = self.sigmoid(out)
        sig_out = sig_out.view(self.batch_size, -1)
        sig_out = sig_out[:, -1]
        
        return sig_out, hidden
    
    def init_hidden(self):
        ''' Hidden state и Cell state инициализируем нулями '''
        # (число слоев; размер батча, размер hidden state)
        h0 = torch.zeros((self.n_layers, self.batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.n_layers, self.batch_size, self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden
    
st.write('''
# Классификация текста NLP моделями
Вашему вниманию представлено три подхода к анализу пользовательских отзывов. Результат работы каждого из них - 
классификация предоставленного текстового фрагмента на русском языке как негативного или позитивного.
Представленная модель называется ... и была обучена ... 
Давайте проверим, как работает каждая из моделей. Введите произвольный текст в поле ниже 
(или вставьте скопированный).
''')
         
output=''       

txt_label = '''Ваш текст:'''
txt = st.text_area(label=txt_label, height=400)
with st.form('key1'):
    button_check = st.form_submit_button("Получить результат")
col1, col2, col3 = st.columns(3)
    
with col1:
    st.write(''' 
        ### Модель TfIdf + CatBoost
        
        Модель TfIdf позволяет получить векторное представление слов на основе того,
        насколько часто слово встречалось в текстах (частотное представление).
        Для построения векторизатора изпользовался набор отзывов о деятельности банковских организаций
        с сайта [HuggingFace](https://huggingface.co/datasets/merkalo-ziri/vsosh2022).
        Модель CatBoost является передовой моделью машинного обучения (работа с табличными данными). 
        Все отзывы из датасета были векторизованы и собраны в таблицу, на которой была обучена модель CatBoost
    ''')

    with open('vectorizer.pkl', 'rb') as fr:
        vec_from_disk = pickle.load(fr)
    with open('model_weight_rf.pkl', 'rb') as fr:
        mdl_from_disk = pickle.load(fr)
    msg_vector = vec_from_disk.transform([txt]) # message_text - сообщение пользователя в виде строки (str)
    msg_df = pd.DataFrame.sparse.from_spmatrix(msg_vector)
    msg_df.columns = vec_from_disk.get_feature_names_out()
    probability = mdl_from_disk.predict_proba(msg_df)[0][1]

    if probability > 0.5:
        output = 'POSITIVE'
    else:
        output = 'NEGATIVE'
    if button_check:
        st.markdown(f'`{output}`')

with col2:
    st.write('''
        ### Модель LSTM
        Данная модель является нейросетевой, рекуррентного типа. Модели такого класса учитывают не только
        частоту появления слова в текстах, но и взаимное распоожение слов.
        Для обучения этой модели использовался тот же самый датасет отзывов о деятельности банковских организаций
        с сайта [HuggingFace](https://huggingface.co/datasets/merkalo-ziri/vsosh2022).
    ''')
    with open('vocab.pkl', 'rb') as fr:
        vocab_to_int = pickle.load(fr)
    model_lstm = sentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)
    model_lstm.load_state_dict(torch.load('lstm_weights.pt'))
    model_lstm.eval()    
    output = model_lstm.to(device)(preprocess_single_string(txt, seq_len=150, vocab_to_int=vocab_to_int).unsqueeze(0).to(device))
    probability = output[0][0]
    if probability > 0.5:
        output = 'POSITIVE'
    else:
        output = 'NEGATIVE'
    
    if button_check:
        st.markdown(f'`{output}`')


with col3:
    st.write('''
        ### Модель семейства BERT
        
        Данная модель является нейросетевой, типа "трансформер". Модели такого класса могут "запоминать" 
        длинные последовательности слов и, как правило, долго обучаются на большом объеме текстов.
        Представленная модель является предобученной, ее описание можно найти 
        [здесь](https://huggingface.co/blanchefort/rubert-base-cased-sentiment)
    ''')
        
    
    API_TOKEN = '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'
    API_URL = "https://api-inference.huggingface.co/models/blanchefort/rubert-base-cased-sentiment"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    if txt != '':
        #txt = '[{}] '.format(5) + txt
        for i in range(10):
            response = query({"inputs":f'{txt}'})

            if type(response) == list:
                output = response[0][0]['label']
                if button_check:
                    st.markdown(f'Отзыв относится к классу: `{output}`')
                break
            
        if type(response) != list:
            if button_check:
                st.markdown(':red[Что-то пошло не так, попробуйте ещё раз]')