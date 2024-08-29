from langchain.agents import create_sql_agent, create_react_agent, Tool, AgentExecutor, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from typing import Annotated
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

if "messages" not in st.session_state :
    st.session_state["messages"] = []

st.set_page_config(
    page_title="Final Code Challange",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        # print(*args)
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        # print(*args)
        save_message(self.message['output'], "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        # self.message_box.markdown(self.message)


# SQL Agent 생성
llm = ChatOpenAI(
    temperature=0.1, 
    model_name='gpt-4o-mini', 
    streaming=True,
    # callbacks=[
    #     ChatCallbackHandler(),
    # ],
)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. and TRANSLATE to korean

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


prompt = PromptTemplate.from_template(template)

st.title("Final Code Challange")

st.markdown(
    """
Welcome!
"""
)

################### Agent TOOL Definition ###################

search = GoogleSerperAPIWrapper()

@tool
def get_db_tool(db_path) -> Tool:
    """영화(Movie)와 관련된 내용을 찾을 때 이 도구를 사용해야 합니다. 이 도구는 데이터베이스를 조회합니다."""
    
    # 데이터베이스 설정
    db = SQLDatabase.from_uri(db_path)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # SQL Agent 생성 
    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    return Tool(
        name="Database_Search_for_movie",
        func=sql_agent.run,
        description="영화 관련 내용을 찾을 때 사용합니다.",
        verbose=True
    )

@tool
def get_pdf_tool(pdf_path) -> Tool:
    """PDF에서 검색해야할 경우 이 도구를 사용해야 합니다"""
    
    # loader = PyPDFLoader("SPRi AI Brief_8월호_산업동향.pdf")
    loader = PyPDFLoader(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = loader.load_and_split(text_splitter)
    
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="'PDF에서 검색해야할 경우 이 도구를 사용해야 합니다",
    )
    
    return tool

repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "Python code to run to generate a chart with streamlit"]
):
    """Use this to execute streamlit python code. If you want to see the output of a value, you should print it out with `streamlit.pyplot()`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

agent_tools = [
    Tool(
        name="Google_Search",
        func=search.run,
        description="웹 검색이 필요할 때 사용합니다.",
        verbose=True
    ),
    get_db_tool("sqlite:///movies.sqlite"),
    get_pdf_tool("/Users/hwangms/Documents/workspace/LLM_Study/langchain_study/hwang/analysis/code_challange/SPRi AI Brief_8월호_산업동향.pdf"),
    python_repl,
]

################### Agent Definition ###################

# React Agent 생성
react_agent = create_react_agent(
    llm, 
    agent_tools, 
    prompt, 
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=agent_tools,
    handle_parsing_errors=True,
    verbose=True,
    # return_intermediate_steps=True,
)


message = st.chat_input("질문해주세요.")
send_message("안녕하세요.", "ai", save=False)
paint_history()

if message:
    send_message(message, "user")
    # with st.chat_message("human"):
    #     st.write(message)
        
    response = agent_executor.invoke({"input":message})
    
    send_message(response['output'], "ai")
    
