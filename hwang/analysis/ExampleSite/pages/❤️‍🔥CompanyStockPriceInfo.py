import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from pykrx import stock 
from datetime import datetime 
import json 


# 최근 주가 정보 가져오기
def get_stock_price(company_name):
    # 오늘 날짜와 이전 거래일을 구합니다.
    today = datetime.now().strftime("%Y%m%d")
    last_day = stock.get_nearest_business_day_in_a_week(today)

    # 상장된 모든 종목의 정보에서 회사명에 해당하는 종목코드 찾기
    # stock_list = stock.get_market_ticker_name()
    stock_list = pd.DataFrame({'code': stock.get_market_ticker_list(today)})
    stock_list['name'] = stock_list['code'].map(lambda x: stock.get_market_ticker_name(x))

    code = stock_list[stock_list['name']==company_name].code

    # 마지막 거래일의 종가 가져오기
    price_data = stock.get_market_ohlcv_by_date(last_day, last_day, code)

    if not price_data.empty:
        price = str(price_data['종가'].iloc[0])
    else :
        price = "주가정보를 가져올 수 없습니다. 상장된 회사가 맞는지 확인해보세요."

    result = {
        "company_name" : company_name,
        "price" : price,
    }
    
    return json.dumps(result, ensure_ascii=False)

functions = [
    {
        "name": "get_stock_price",
        "description": "질문에서 회사명을 찾아 해당 회사의 주가정보를 반환해주는 역할을 하는 함수입니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "한화시스템, 삼성전자, 네이버, 카카오",
                }
            },
            "required": ["company_name"],
        },
    }
]

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
).bind(
    function_call={
        # "name": "get_stock_price",
        "auto",
    },
    functions=[
        functions,
    ],
)

st.set_page_config(
    page_title="회사 주가정보를 가져오자!!",
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


final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            어떤 회사의 주가정보를 알려달라고 하면 해당 회사의 가장 최근 주가정보를 알려줘야 합니다.
            
            ------
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("🤖 회사 주가정보를 알려주는 챗봇 만들기")

st.markdown("""
💥 **Mission** :  Function Calling 사용!!
""")

paint_history()

message = st.chat_input("질문해주세요.")

if message:
    send_message(message, "Human")
    chain = {"question": RunnablePassthrough()} | final_prompt | llm #| output_parser
    
    with st.chat_message("ai"):
        chain.invoke(message)
        
else:
    st.session_state["messages"] = []





# prompt = PromptTemplate.from_template("Make a quiz about {city}")
# chain = prompt | llm
# response = chain.invoke({"city": "rome"})
# response = response.additional_kwargs["function_call"]["arguments"]
# response