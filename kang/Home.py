import streamlit as st
from datetime import datetime

today = datetime.today().strftime('%H:%M:%S')

st.title(today)

st.selectbox('Select', ['A', 'B', 'C'])

value = st.slider("temperature", 0, 100, 25)