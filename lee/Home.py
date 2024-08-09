import streamlit as st
from langchain.prompts import PromptTemplate


## 무엇을 넘겨주던 그것에 맞는 UI를 생성할 수 있는 기능
##st.write("hello")
##st.write([1, 2, 3, 4])
## st.write(PromptTemplate)

## Streamlit 에서 sidebar를 만들 때의 패턴이 있음


###자주 사용하지는 않는 방법
###st.sidebar.title("sidebar title")
###st.sidebar.text_input("xxx")
###"""

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🐱‍👓",

)
st.title("FullstackGPT Home")


## Markdown으로 페이지 구성이 가능
st.markdown(
    """
    # Hello!

    Welcome to my FullstackGPT Portfolio!

    Here are the apps I made:

    - [ ] [DocumentGPT](/DocumentGPT) 
    - [ ] [PrivateGPT](/PrivateGPT) 
    - [ ] [QuizGPT](/QuizGPT) 

    """
)