import streamlit as st
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage.file_system import LocalFileStore
from langchain.embeddings.cache import CacheBackedEmbeddings


st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}") 
    spliter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=200,
        chunk_overlap=50,
    )
    loader = UnstructuredFileLoader(file_path) 
    docs = loader.load_and_split(text_splitter=spliter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(documents=docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

st.title("DocumentGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about your files.
""")

file = st.file_uploader("Upload a .txt .pdf or .docx file", 
                        type=["txt", "pdf", "doc"],
                        )


if file:
    retriever = embed_file(file)
    s = retriever.invoke("winston")
    st.write(s)