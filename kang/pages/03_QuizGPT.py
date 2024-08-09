import streamlit as st
from langchain.retrievers.wikipedia import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="Quize GPT Home",
    page_icon="‼️",
)

st.title("Quiz GPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "AI")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
        ],
    )

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    spliter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=200,
        chunk_overlap=50,
    )
    loader = UnstructuredFileLoader(file_path) 
    docs = loader.load_and_split(text_splitter=spliter)
    return docs

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


with st.sidebar:
    docs = None
    choice = st.selectbox("Choose what you want to use", 
                          ("File", "Wikipedia Article",),
                          )
    if choice == "File":
        file = st.file_uploader("Upload a file",
                                type=["txt", "pdf"])
        if file:
            with st.status("Loading file..."):
                docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            retriever = WikipediaRetriever(top_k_results=3)
            with st.status("Searching Wikipedia"):
                docs = retriever.get_relevant_documents(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    st.write(docs)
