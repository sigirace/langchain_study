import streamlit as st
# from datetime import datetime
# # from langchain.prompts import PromptTemplate
# # st.title("hello world")
# # st.subheader("welcome to streamlit")
# # st.markdown("""
# #     ### I love it
# # """)

# # st.write([1, 2, 3, 4])
# # # st.write(PromptTemplate)
# # p = PromptTemplate.from_template("xxxx")
# # st.write(p)

# # st.selectbox(
# #     "Choose your model",
# #     (
# #         "GPT-3",
# #         "GPT-4",
# #     ),
# # )

# today = datetime.today().strftime("%H:%M:%S")

# st.title(today)


# model = st.selectbox(
#     "Choose your model",
#     (
#         "GPT-3",
#         "GPT-4",
#     ),
# )

# if model == "GPT-3":
#     st.write("cheap")
# else:
#     st.write("not cheap")
#     name = st.text_input("What is your name?")
#     st.write(name)

#     value = st.slider(
#         "temperature",
#         min_value=0.0,
#         max_value=1.0,
#     )

#     st.write(value)

# tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])
# with tab_one:
#     st.write('a')
# with tab_two:
#     st.write('b')
# with tab_three:
#     st.write('c')

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

st.markdown(
    """
# Hello!

Welcome to my FullstackGPT Portfolio!

Here are the apps I made:

- [ ] [DocumentGPT](/DocumentGPT)
- [ ] [PrivateGPT](/PrivateGPT)
- [ ] [QuizGPT](/QuizGPT)
- [ ] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
"""
)
