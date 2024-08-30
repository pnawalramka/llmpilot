import streamlit as st

from llm import LLMPilot


st.title('LLM Pilot')
llm = LLMPilot()

with st.form('llm_input_form'):
    text = st.text_area('Enter text:', 'Who is Tux?')
    submitted = st.form_submit_button('Submit')

    if submitted:
        res = llm.llm_response(text)
        st.info(res)
