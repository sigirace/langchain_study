from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

base_url = "https://www.fmkorea.com"

html2text_transformer = Html2TextTransformer()

@st.cache_data(show_spinner="Loading website...")
def load_website(id_list):
    docs = []
    for id in id_list:
        loader = AsyncChromiumLoader([base_url + "/" + id.strip()])
        doc = loader.load()
        # HTML 파싱
        soup = BeautifulSoup(doc[0].page_content, 'html.parser')
        
        # header와 footer 제거
        if soup.header:
            soup.header.decompose()
        if soup.footer:
            soup.footer.decompose()
        
        str(soup.get_text())

    return docs

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    id_list = st.text_input(
        "Write down a URL",
        placeholder="Doc id",
    )

if id_list:
    id_list = id_list.split(",")
    docs = load_website(id_list)
    st.write(docs)
