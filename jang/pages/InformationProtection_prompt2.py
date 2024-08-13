import streamlit as st
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from settings import getLlm

st.set_page_config(
    page_title="개인 정보를 보호하자!!",
    page_icon="🤖",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "count" not in st.session_state:
    st.session_state["count"] = 0


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


def get_count():
    return st.session_state["count"]


st.title("🤖 개인 정보 절대 보호")

st.markdown("""

안녕하세요~~            

이번 챌린지는 사용자 질문에 따라 개인 정보를 위반하고 있는지 파악하는 챗봇을 구축하는 것입니다.
""")


firtst_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            너는 테스트용 챗봇이야.
            human message에 개인정보형식(주민등록번호, 이메일주소, 전화번호 등)이 포함되어 있을때는 '개인정보'라고 답해줘
            """,
        ),
        ("human", "{message}"),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            너는 테스트용 챗봇이야.
            context에 개인정보라고 적혀있으면 아래 예시에 맞춰 답해주고 아니면 알아서 대답해줘
            예시)'개인정보가 검출되었습니다.\n\n 개인정보 유형 : \n\n 검출정보 :'
            context:{context}
            """,
        ),
        ("human", "{message}"),
    ]
)

message = st.chat_input("ask me anything!!")
send_message("위반사항만 아니면 대답해 드립니다.", "ai", save=False)
paint_history()
if message:
    send_message(message, "human")
    first_chain = (message | getLlm())
    # final_chain = {"context": first_chain, "question": RunnablePassthrough()
    #                } | final_prompt | getLlm()
    # response = final_chain.invoke({"message": message})
    response = first_chain.invoke({"message": message})
    if response.content.startswith("개인정보가 검출되었습니다"):
        st.session_state["count"] += 1
    with st.sidebar:
        st.write(f"개인정보 위반 횟수 : {get_count()}건")
    send_message(response.content, "ai")
