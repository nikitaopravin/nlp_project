import streamlit as st
import requests
import time

st.write('''
# NLP модель TextSummary

Класс данных моделей предназначен для получения сводной информации о тексте. 

Передаем текст в модель - и она пишет нам, какая основная мысль, идея данного текста. 
То есть предоставляет нам краткую сводную информацию, или резюме по тексту.

Представленная модель называется ... и была обучена ... 

Давайте проверим, как она работает! Введите произвольный текст в поле ниже 
(или вставьте скопированный). Максимальная длина - ххх
''')

txt_label = '''Ваш текст:'''

txt_value = ''

txt = st.text_area(label=txt_label, value=txt_value, height=400)

API_TOKEN = ''

API_URL = "https://api-inference.huggingface.co/models/cointegrated/rut5-base-absum"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

if txt != '':
    
    for i in range(10):
        response = query({"inputs":f'{txt}'})
        st.write(response)
        if '0' not in response.keys:
            st.markdown(":red[Ждем ответа модели..]")
            time.sleep(2)
            
        else:
            output = response[0]['summary_text']
            st.markdown(f'Резюме модели: `{output}`')
            break  

    