import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseOutputParser, output_parser

count = 0

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        if "위반" in content

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# class ScoreOutputParser(BaseOutputParser):
#     def parse(self, text):
#         if text.find("위반") > 0 :
#             count = count + 1 
        
#         return count


# output_parser = ScoreOutputParser()

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

st.set_page_config(
    page_title="개인 정보를 보호하자!!",
    page_icon="🤖",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

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

ind_check_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """            
            개인정보 형태는 아래와 같습니다.
            1. 주민등록번호 : 숫자 13자리로 구성되며, '-'문자가 중간에 있을 수 있습니다. 
            2. 이메일 : 영문 텍스트로 되어 있으며 '@'와 도메인이 포함되어 있습니다.
            
            개인정보가 있다면 "위반"이라는 결과도 함께 주세요.
            -------
            """,
        ),
        ("human", "{question}"),
    ]
)

ind_check_chain = ind_check_prompt | llm
# ind_check_chain.invoke({"question": RunnablePassthrough()})

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            질문 내용에 개인정보가 있는지 확인하고, 개인정보가 있다면 어떤 것이 위반인지 답변을 주어야 합니다.
            모르는 질문에는 답변하지 마세요.
            
            ------
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("🤖 개인 정보 절대 보호")

st.markdown("""

안녕하세요~~            

이번 챌린지는 사용자 질문에 따라 개인 정보를 위반하고 있는지 파악하는 챗봇을 구축하는 것입니다.

💥 **Mission** : Chain을 2개 사용해보기!!
""")

with st.sidebar:
    st.write("개인정보 위반 횟수 :", f"{count}건")

send_message("위반사항만 아니면 대답해 드립니다.", "ai", save=False)
paint_history()

message = st.chat_input("질문해주세요.")

if message:
    send_message(message, "Human")
    chain = {"context": ind_check_chain, "question": RunnablePassthrough()} | final_prompt | llm #| output_parser
    
    with st.chat_message("ai"):
        chain.invoke(message)


else:
    st.session_state["messages"] = []