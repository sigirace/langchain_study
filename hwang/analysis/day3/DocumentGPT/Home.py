### 7.2 Data Flow

# import streamlit as st
# from langchain.prompts import PromptTemplate
# from datetime import datetime

# today = datetime.today().strftime("%H:%M:%S")

# st.title(today)

# model = st.selectbox("Choose your model", ("GPT-3", "GPT-4"))

# if model == "GPT-3":
#     st.write("cheap")
# else:
#     st.write("not cheap")

# st.write(model)

# name = st.text_input("What is your name?")

# st.write(name)

# value = st.slider("temperature", min_value=0.1, max_value=1.0)

# st.write(value)



### 7.3 Multi Page

# import streamlit as st

# st.title("title")

# with st.sidebar:
#     st.title("sidebar title")
#     st.text_input("xxx")


# tab1, tab2, tab3 = st.tabs(["A", "B", "C"])

# with tab1:
#     st.write("a")

# with tab2:
#     st.write("b")

# with tab3:
#     st.write("c")



import streamlit as st

st.set_page_config(
    page_title = "FullstackGPT Home",
    page_icon = "😎",
)

st.markdown(
    """
# Hello!
            
Welcome to my FullstackGPT Portfolio!
            
Here are the apps I made:
            
- [x] [DocumentGPT](/DocumentGPT)
- [x] [PrivateGPT](/PrivateGPT)
- [x] [QuizGPT](/QuizGPT)
- [ ] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
"""
)