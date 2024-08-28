import streamlit as st

from llm import get_llm, llm_response

st.title('LLM Pilot')

llm = get_llm()

with st.form('llm_input_form'):
    text = st.text_area('Enter text:', 'Who is Tux?')
    submitted = st.form_submit_button('Submit')

    if submitted:
        res = llm_response(llm, text)
        st.info(res)
