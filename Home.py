import streamlit as st
import requests
from streamlit_lottie import st_lottie



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://catherineasquithgallery.com/uploads/posts/2021-02/1612805483_131-p-myagkii-fon-goluboi-210.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


st.write("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fascinate');
html, body, [class*="css"]  {
   font-family: 'Sono', cursive;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## Классификация отзыва на фильм на русском языке")
st.markdown("### Разрабатка multipage-приложения с использованием streamlit:")



def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_tno6cg2w.json')
lottie_coding_1 = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_tfb3estd.json')
lottie_coding_2 = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_zmlg2tee.json')

with st.container():
    image_col, text_col = st.columns((1,2))
    with image_col:
        st_lottie(lottie_coding, height=200)

    with text_col:
        st.markdown("##### Результаты предсказаний класса (позитивный/негативный) тремя моделями:")
        st.write("""
            - Классический ML-алгоритм, обученный на TF-IDF представлении;
            - LSTM модель;
            - BERT.
            """)

with st.container():
    image_col, text_col = st.columns((1,2))
    with image_col:
        st_lottie(lottie_coding_1, height=200)

    with text_col:
        st.markdown("#### Генерация текста GPT-моделью по пользовательскому prompt:")
        st.write("""
            - Пользователь может регулировать длину выдаваемой последовательности;
            - Число генераций;
            - Температуру или top-k/p.
            """)
        

with st.container():
    image_col, text_col = st.columns((1,2))
    with image_col:
        st_lottie(lottie_coding_2, height=200)

    with text_col:
        st.markdown("#### Произвольная задача (используем подходящие предобученные модели):")
        st.write("""
            - Саммаризация текста: пользователь вводит большой текст, модель делает саммари.
            """)