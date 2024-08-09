import time
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

# with st.chat_message("human"):
#     st.write("Hello")
# with st.chat_message("ai"):
#     st.write("How R U")

# with st.status("embedding File..", expanded=True) as status:
#     time.sleep(1)
#     st.write("getting the file")
#     time.sleep(1)
#     st.write("embedding the file")
#     time.sleep(1)
#     st.write("chaching the file")
#     status.update(label="Error", status="error")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False,
    )


message = st.chat_input("Send a message to the ai ")

if message:
    send_message(message, "human")
    time.sleep(1)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)
