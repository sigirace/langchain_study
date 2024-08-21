import streamlit as st
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from settings import getLlm
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI,AzureOpenAI
import os

st.set_page_config(
    page_title="functionCall",
    page_icon="🤖",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

def get_stock_price(stock_name, stock_price):
    stock_info = {
        "stock_name": stock_name,
        "stock_price": stock_price,
    }
    return stock_info


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


st.title("🤖 FunctionCall")

st.markdown("""

FunctionCall
""")

prompt = ChatPromptTemplate.from_messages([("human", "{message}"),])
second_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            너는 주식 가격을 알려주는 챗봇이야.
            human message에 object를 아래 예시처럼 정리해서 알려줘
            예시 )'말씀하신 주식의 가격은 아래와 같습니다.\n\n 주식명 : 삼성전자 \n\n 주식가격: 80000'
            """,
        ),
        ("human", "{message}"),
    ]
)

functions = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price in a given stock_name",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_name": {
                    "type": "string",
                    "description": "The stock name, e.g. 한화시스템,삼성전자,SK하이닉스",
                },
                "stock_price": {
                    "type": "string",
                    "description": "The stock price, e.g. 19000,86000,170000",
                },
            },
            "required": ["stock_name", "stock_price"],
        },
    }
]


message = st.chat_input("ask me anything!!")
send_message("채팅 시작", "ai", save=False)
paint_history()
if message:
    send_message(message, "human")

    llm=AzureChatOpenAI(
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("ENDPOINT"),
        azure_deployment=os.getenv("CHAT_MODEL"),
        api_key=os.getenv("API_KEY"),
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    chain = (prompt | llm.bind(
        function_call={
            "name": "get_stock_price",
        },
        functions=functions,
    ))
    response = chain.invoke(message)
    if response.additional_kwargs["function_call"] :
        print(response.additional_kwargs["function_call"])
    second_response = eval(
        response.additional_kwargs["function_call"]["arguments"])
    # st.write(second_response)
    second_message = get_stock_price(
        second_response.get("stock_name"), second_response.get("stock_price"))
    # st.write(second_message)

    # second_chain = (second_prompt | getLlm())
    second_chain = (second_prompt | llm)
    # st.write(second_chain.invoke({"message": [second_message]}))
    second_response =second_chain.invoke({"message": [second_message]})
    # print(second_response)
    send_message(second_response.content, "ai")
