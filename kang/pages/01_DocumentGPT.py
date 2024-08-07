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

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_data(show_spinner="Embedding file...")
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

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


st.title("DocumentGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about your files.
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", 
                            type=["txt", "pdf", "doc"],
                            )


if file:
    retriever = embed_file(file)
    send_message("I`m ready! ask me about the file", "ai", save=False)
    paint_history()
    message = st.chat_input("you can ask me about the file")
    if message:
        send_message(message, "Human")
else:
    st.session_state["messages"] = []