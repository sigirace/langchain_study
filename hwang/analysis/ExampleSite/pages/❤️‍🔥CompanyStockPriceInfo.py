import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from pykrx import stock 
from datetime import datetime 
import json
import pandas as pd 
from langchain.callbacks.base import BaseCallbackHandler
import openai


# 최근 주가 정보 가져오기
def get_stock_price(company_name):
    print("hwangms_get_stock_price")
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


st.set_page_config(
    page_title="회사 주가정보를 가져오자!!",
    page_icon="🤖",
)

st.title("🤖 회사 주가정보를 알려주는 챗봇 만들기")

st.markdown("""
💥 **Mission** :  Function Calling 사용!!
""")


function = [
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

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 펀드매니저 입니다.
            """,
        ),
        ("human", "{question}"),
    ]
)

message = st.chat_input("질문해주세요.")

if message:
    with st.chat_message("user"):
        st.write(message)
        
    messages = [{"role": "user", "content": message}]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=function,
        function_call="auto",
        )
    
    response_message = response["choices"][0]["message"]
    
    if response_message.get("function_call"):
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_stock_price": get_stock_price,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            company_name=function_args.get("company_name"),
        )

        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        
        second_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
        )  # get a new response from GPT where it can see the function response

    with st.chat_message("ai"):
        st.markdown(second_response.choices[0].message.content)

