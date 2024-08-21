import os
from langchain_openai import AzureChatOpenAI


class Settings():
    ENDPOINT: str = 'ENDPOINT'
    API_KEY: str = 'API_KEY'
    API_VERSION: str = 'API_VERSION'
    CHAT_MODEL: str = 'CHAT_MODEL'
    EMBEDDING_MODEL: str = 'EMBEDDING_MODEL'
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_ENDPOINT: str = 'LANGCHAIN_ENDPOINT'
    LANGCHAIN_API_KEY: str = 'LANGCHAIN_API_KEY'

    class Config:
        env_file = '.env'


def getLlm():
    return AzureChatOpenAI(
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("ENDPOINT"),
        azure_deployment=os.getenv("CHAT_MODEL"),
        api_key=os.getenv("API_KEY"),
        streaming=True,
    )
