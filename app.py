import os

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import streamlit as st


st.title('LLM Pilot')
openai_api_key = os.getenv('OPENAI_API_KEY')
model_choice = os.getenv('MODEL_CHOICE')

if not model_choice or model_choice == 'gpt4':
    llm = ChatOpenAI(
        model='gpt-4-turbo', temperature=0.7, api_key=openai_api_key)    
elif model_choice == 'gpt3':
    llm = OpenAI(
        model='gpt-3.5-turbo-instruct', temperature=0.7, api_key=openai_api_key)


def llm_response(input_text):
    llm_res = llm.invoke(input_text)
    print(llm_res)
    return llm_res.content if isinstance(llm, ChatOpenAI) else llm_res


with st.form('llm_input_form'):
    text = st.text_area('Enter text:', 'Who is Tux?')
    submitted = st.form_submit_button('Submit')

    if submitted and openai_api_key.startswith('sk-'):
        res = llm_response(text)
        st.info(res)
