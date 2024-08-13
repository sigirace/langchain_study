import streamlit as st

from langchain.schema.messages import AIMessageChunk

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


st.title("🤖 개인 정보 절대 보호")

st.markdown("""

안녕하세요~~            

이번 챌린지는 사용자 질문에 따라 개인 정보를 위반하고 있는지 파악하는 챗봇을 구축하는 것입니다.

💥 **Mission** : Chain을 2개 사용해보기!!
""")

with st.sidebar:
    st.write("개인정보 위반 횟수 :", "n건")

send_message("위반사항만 아니면 대답해 드립니다.", "AI", save=False)
paint_history()

message = st.chat_input("you can ask me about the file")

if message:
    send_message(message, "Human")        
    
    with st.chat_message("AI"):
        st.markdown("아직구현되지 않았음")
        
        