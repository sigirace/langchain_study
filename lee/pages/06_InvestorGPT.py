import streamlit as st
import time

import os
os.environ["TIKTOKEN_CACHE_DIR"] = './etc'

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.agents import initialize_agent, AgentType, load_tools   
from langchain.tools import StructuredTool, Tool, BaseTool
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Any, Type
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from pydub import AudioSegment
import math, requests
import subprocess
import glob, openai


## [Langchain Setting]=============================================================== ##

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
)

llm = ChatOpenAI(temperature=0.1)

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

st.set_page_config(
    page_title = "InvestorGPT",
    page_icon = "🖥️",
)

st.title("InvestorGPT")



## [Functions and Agents]======================================================================= ## 

class stockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")

class stockMarketSymbolSearchTool(BaseTool):
    name = "stockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    Example query : Stock Market Symbol for Apple Company
    """

    args_schema : Type[stockMarketSymbolSearchToolArgsSchema] = stockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)



## Tool Schema : str input (Symbol of a company)
class CompanyOverviewSchema(BaseModel):
    symbol: str = Field(description="Symbol of the company. Example: APPL, TSLA")

class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = "Use this tool to get an overview of the financials of the company. You should enter a stock symbol."
    args_schema : Type[CompanyOverviewSchema] = CompanyOverviewSchema
    
    def _run(self, symbol):
        ddg = DuckDuckGoSearchAPIWrapper()
        r = requests.get("https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()

## Tool Implementation
class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = "Use this tool to get an income state of the financials of the company. You should enter a stock symbol."
    args_schema : Type[CompanyOverviewSchema] = CompanyOverviewSchema
    
    def _run(self, symbol):
        ddg = DuckDuckGoSearchAPIWrapper()
        r = requests.get("https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()["annualReports"]
    
class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description = "Use this tool to get the weekly performance of a company stock. You should enter a stock symbol."
    args_schema : Type[CompanyOverviewSchema] = CompanyOverviewSchema
    
    def _run(self, symbol):
        ddg = DuckDuckGoSearchAPIWrapper()
        r = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}")
        response = r.json()
        return list(response["Time Series (Digital Currency Weekly)"].items())[:100]
    

agent = initialize_agent(
    llm = llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        stockMarketSymbolSearchTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs={
        "system_message":SystemMessage(
            content="""
            You are a hedge fund manager.
            
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            
            Consider the performance of a stock, the company overview and the income statement.
            And also you have to show it you found.
            
            Be assertive in your judgement and recommend the stock or advise the user against it.
        """
        )
    }
)



company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent.invoke(company)
    st.write(result["output"])