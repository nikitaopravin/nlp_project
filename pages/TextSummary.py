import streamlit as st
import requests

st.write('''
# NLP модель TextSummary

Класс данных моделей предназначен для получения сводной информации о тексте. 

Передаем текст в модель - и она пишет нам, какая основная мысль, идея данного текста. 
То есть предоставляет нам краткую сводную информацию, или резюме по тексту.

Представленная модель называется 
[cointegrated/rut5-base-absum](https://huggingface.co/cointegrated/rut5-base-absum) и она основыввется на модели
[Google's mT5](https://github.com/google-research/multilingual-t5), дообученнной на русских текстах.

Давайте проверим, как она работает! 

*Введите произвольный текст в поле ниже (или вставьте скопированный).*
''')

txt_label = '''Ваш текст:'''

SUM_TOKEN = st.secrets["SUM_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/cointegrated/rut5-base-absum"
headers = {"Authorization": f"Bearer {SUM_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def summarize(txt):
    if txt != '':
        
        txt = '[{}] '.format(5) + txt  #проверить!
        for i in range(10):
            response = query({"inputs":f'{txt}', "parameters": {"num_beams": 4,
                          "do_sample" :False, "repetition_penalty": 10.0}})
                    
            if type(response) == list:
                
                output = response[0]['summary_text']
                if '==Rel="nofollow' in output:
                    st.markdown(':red[Что-то пошло не так, попробуйте ещё раз]')
                else:
                    st.markdown(f'Резюме модели: `{output}`')
                break
        
        if type(response) != list:
            st.markdown(':red[Долго ждем ответа модели... попробуйте ещё раз]')
    else:
        st.markdown(f'Резюме модели: `Вы немногословны. Напишите хоть что-нибудь `')   

txt = st.text_area(label=txt_label, height=400)
ret = st.button('Резюмировать!',  type="secondary")

if ret:
     summarize(txt)    