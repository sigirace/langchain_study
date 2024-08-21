
import streamlit as st

import os
os.environ["TIKTOKEN_CACHE_DIR"] = './etc'

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore

from bs4 import BeautifulSoup
import requests
import json

## [Goal]
## GPT에게 특정 회사의 주가를 물어봤을 때, 학습되지 않은 최신 정보를 대답하지 못할 수 있다.
## 이 경우 GPT가 Optional Function Callback을 통해 올바른 정보를 가져오도록 구현해 보자

## [Initialization]


FILE_NAME = "./files/code_data.txt"

## function call setting
function = {
    "name":"get_stockPrice",
    "description":"특정 기업 종목코드를 입력받아 그 주가를 확인해 값을 반환하는 함수입니다.",
    "parameters":{
        "type":"object",
        "properties":{
            "code":{
                "type":"string",
                "description":"KOSPI, KOSDAC에 대응하는 종목코드"
            }
        }
    },
    "required":["code"]
}


llm = ChatOpenAI(
    temperature=0.1,
).bind(
    function_call="auto",
    functions=[function]
)

llm_sub = ChatOpenAI(temperature=0.1)

## [Functions]

@st.cache_data(show_spinner="Embedding...")
def embed_file(file_name):
    loader = UnstructuredFileLoader(file_name)
    cache_dir = LocalFileStore("./.cache/embeddings/stock")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    embedding = OpenAIEmbeddings()
    docs = loader.load_and_split(text_splitter=splitter)    

    cached_embedding = CacheBackedEmbeddings.from_bytes_store(embedding, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embedding)

    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        message
    if save:
        save_message(message, role)

def paint_message():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)



## Function call (Input : str, Output : str)
def get_stockPrice(code):
    
    print("function called... : " + code)
    url_target = "https://finance.naver.com/item/main.naver?code=" + code
    print(url_target)
    response = requests.get(url_target)
    #st.write(response.text)

    soup = BeautifulSoup(response.text, 'html.parser')
    result = soup.select_one('#middle > dl > dd:nth-child(5)')
    return result.text

## [Prompts]

## 기업명 입력 시 잘못된 종목코드를 들고 오거나, 존재하지 않는 기업명 또는 코드를 답변하는 경우가 발생
## 따라서 RAG로 국내 상장된 대표적인 기업 일부를 넣어 주며,
## '하이닉스', '에어로' 등의 일부 키워드 검색에도 답변할 수 있도록 프롬프트 설계
stockCodePrompt = ChatPromptTemplate.from_messages({
    ("system",
    """
    당신은 질문으로 주어지는 기업명에 해당하는 KOSPI, KOSDAC 주식 종목코드를 알려줍니다.
    반드시 존재하는 기업명에 한해 한해 종목코드만을 답변합니다.
    
    """),
    ("human","{question}")
})

stockPricePrompt = ChatPromptTemplate.from_messages({
    ("system",
    """
    당신은 질문의 정보를 사람이 읽기 좋게 재구성하는 역할을 수행합니다.
    최대한 자연스러운 문장으로 바꾸어 답변합니다.

    """),
    ("human","{question}")
})

## [Functions_logic]

## Input : dict(context(retriever), question(str)) / Output : str
def get_codes(question):

    #question = inputs["question"]   ## String
    #context = inputs["context"]     ## List of Documents

    ## Convert context to a single string (for chain input)
    # context_str = ""
    # for doc in context : 
    #     context_str.join(doc.page_content)

    ## chain의 결과는 single string이다.
    chain = stockCodePrompt | llm
    response = chain.invoke(question)

    #st.write(response)

    return {"context":response, "question":question}

def get_price(inputs):

    question = inputs["question"]
    context = inputs["context"]

    if context.additional_kwargs["function_call"]:
        data = json.loads(context.additional_kwargs["function_call"]["arguments"])## "code":"272210"
        code = data.get("code")
        price_hint = get_stockPrice(code)
        
        price_hint = "종목코드 {0} : {1}".format(code, price_hint)
        #st.write(price_hint)
        
        chain = {"question":RunnablePassthrough()} | stockPricePrompt | llm_sub
        response = chain.invoke(price_hint)
        return response.content

    else:
        chain = stockPricePrompt | llm_sub
        response = chain.invoke(question)

    return response.content


## [Views_Streamlit]
st.title("StockPriceGPT")
message = st.chat_input("정보를 알고 싶은 기업명을 입력하세요")

send_message("안녕하세요! 주가를 알고 싶은 회사 명을 알려주세요!", "ai", False)

if "messages" not in st.session_state:
    st.session_state["messages"]=[]

## [Logics]


if message:
    send_message(message, "human", True)

    # retriever = embed_file(FILE_NAME)

    chain = {"question":RunnablePassthrough()} | RunnableLambda(get_codes) | RunnableLambda(get_price)
    response = chain.invoke(message)

    ## response_vanila = llm.predict(message)

    send_message(response, "ai", True)
    #send_message(response_vanila, "ai", True)
