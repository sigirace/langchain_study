
import streamlit as st
import time

import os
os.environ["TIKTOKEN_CACHE_DIR"] = './etc'

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.prompts import ChatPromptTemplate

from langchain.callbacks import StreamingStdOutCallbackHandler

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

## [Langchain Setting]=============================================================== ##

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
            
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token:str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

        

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

## ================================================================================== ##

## [Functions]=================================================================== ##

## decorator를 넣어 file이 cache에 있는 항목과 동일하다면 실행하지 않고 jump함
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    ## 가져온 파일을 langchain의 loader에게 넘겨주어야 함
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    ##st.write(file_content, file_path)

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    
    loader = UnstructuredFileLoader("./files/document.txt")
    docs = loader.load_and_split(text_splitter=splitter)

    cache_dir = LocalFileStore("./cache/embeddings/{file_name}")
    
    embedding = OpenAIEmbeddings()
    cached_embedding = CacheBackedEmbeddings.from_bytes_store(embedding, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embedding)
    retriever = vectorstore.as_retriever()
    return retriever

## 추가 메시지를 보내면 log가 추가되지 않고 기존 대화가 대체되어 버림
## message를 보관하는 곳이 없기 때문임 (매 save마다 rerun되기 때문)
## 데이터 보존을 위해 Streamlit의 Session state를 사용함
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message":message, "role":role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)  ##\n\n은 separator이다.

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

## ================================================================================== ##

## [Prompts]========================================================================= ##

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

    Context: {context}
    """,
    ),
    ("human", "{question}"),
])

## ================================================================================== ##

st.set_page_config(
    page_title="DocumentGPT",
)

st.title("Document GPT")

st.markdown(
    """
    Use this chatbot to ask question to an AI about your files!

    Upload your files on the sidebar
    """
)

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)

    paint_history()

    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        chain = {"context":retriever | RunnableLambda(format_docs), 
                 "question":RunnablePassthrough()} | prompt | llm
        
        with st.chat_message("ai"):
            response = chain.invoke(message)

        

else:
    st.session_state["messages"]=[]


##=============================================================================

