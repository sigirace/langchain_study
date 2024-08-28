from langchain.schema import SystemMessage
import streamlit as st
from langchain.prompts import ChatPromptTemplate
import os
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOllama
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import json

import openai


KIPRIS_REST_AccessKey = "sNuGpLPQtMYqZsEabQbu466205ZRjzXCDxgLpZMRdes="
userId=""
password=""

# 특허·실용 공개·등록공보
Publication_Gazette_URL = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getWordSearch?word=센서&year=0&ServiceKey="

class Publication_GazetteArgsSchema(BaseModel):
    word: str = Field(
        description="검색단어, 예)센서",
    )
    year: str = Field(
        description="검색년도 범위, 예) 0~10 사이의 값중 하나를 가지며 올해는 0, 작년은 1, 재작년은 2순으로 숫자가 커짐",
    )


class Publication_GazetteTool(BaseTool):
    name = "Publication_Gazette"
    description = """
    한국 특허 / 실용신안 공개 및 등록공보를 얻고자 할 때 사용.
    검색단어와 검색년도 범위를 입력해야 함.
    api를 통해 검색된 데이터 내에서만 답변해야하며 제시된 kipris api외에 다른 사이트 api를 참고하면 안됨
    """
    args_schema: Type[Publication_GazetteArgsSchema] = Publication_GazetteArgsSchema

    def _run(self, word, year):
        r = requests.get(
            f"http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getWordSearch?word={word}&year={year}&ServiceKey={KIPRIS_REST_AccessKey}"
        )
        r_decode = r.content.decode('utf-8')
        return r_decode
    



class Similar_SearchToolArgsSchema(BaseModel):
    keyword: str = Field(
        description="사용자 질의내용 중 핵심 키워드. 사용자가 한글로 입력하면 한글을 우선하여 찾는다.",
    )

class RelateComment_SearchToolArgsSchema(BaseModel):
    articleId: str = Field(
        description="게시글을 구분할 수 있는 pk이며 게시글과 연관된 답글(Comment)을 찾고자 할 때 articleId은 답글(Comment)에서 fk로 잡혀있어서 해당 필드 값 기준으로 연관된 답글(Comment)를 찾을 수 있다.",
)


class BoardArticle_SearchTool(BaseTool):
    name = "BoardArticle_Search"
    description = """    
    모든 입력은 한글로 처리하며, 번역하지 말고 그대로 사용하세요.
    찾은 답글도 번역하지 않도록 한다.
    검색단어(keyword)를 입력해야 함.
    """
    
    args_schema: Type[Similar_SearchToolArgsSchema] = Similar_SearchToolArgsSchema

    def loginCleverse(self):
        # 1. token 가져오기
        login_url = "https://hsi.cleverse.kr/api/auth/authenticate"
        login_headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }

        login_temp = {
            "userId":userId,
            "password":password,
            "domain":"hsi.cleverse.kr"
        }

        login_data = json.dumps(login_temp)

        login_response = requests.post(login_url, headers=login_headers, data=login_data)

        login_response_json = login_response.json()

        login_token = login_response_json.get("result").get("accessToken")
        
        return login_token
        
    def _run(self, keyword):
        articleList_url = "https://hsi.cleverse.kr/api/bbs/article/articleList"

        login_token = self.loginCleverse()
        Authorization_text = "Bearer " + login_token#Bearer가 포함된 accessToken 입력
        articleList_headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Authorization": Authorization_text
        }
        # data
        articleList_temp = {
            "page":1,
            "size":10, # 10000건 추출
            "boardMode":"COMPANY",
            "boardId":125477,  #125477:사이다 / 19662: 공지사항
            "boardType":1,
            "searchType":"SUBJECT",
            "searchKeyword":keyword,
            "order":"ALL",
            "sort":"desc",
            "clubId":0,
            "unreadYn":"N",
            "noticeYn":"N"
        }

        # 딕셔너리를 JSON으로 변환 
        articleList_data = json.dumps(articleList_temp)

        articleList_response = requests.post(articleList_url, headers=articleList_headers, data=articleList_data)

        articleList_response.encoding = 'utf-8'

        articleList_response_json = articleList_response.json()
        articleList_response_df = pd.DataFrame(articleList_response_json)
        articles = articleList_response_df.result.listData['content']
        
        articleTable = pd.DataFrame(columns=['articleId', 'subject', 'contentSummary', 'refCnt', 'nickName', 'boardId'])

        for i, aiticles_data in enumerate(articles) :
            articleTable = pd.concat([articleTable, pd.DataFrame([{'articleId' : aiticles_data['articleId'], 
                                                                'subject' : aiticles_data['subject'], 
                                                                'contentSummary' : aiticles_data['contentSummary'], 
                                                                'refCnt' : aiticles_data['refCnt'],
                                                                'nickName' : aiticles_data['nickName'], 
                                                                'boardId' : aiticles_data['boardId']}])])
        
        return articleTable


class BoardComment_SearchTool(BaseTool):
    name = "BoardComment_Search"
    description = """
    모든 입력은 한글로 처리하며, 번역하지 말고 그대로 사용하세요.
    찾은 답글도 번역하지 않도록 한다.
    그룹웨어 게시판에서 게시글 정보에 대한 답변(comment)를 얻고자 할 때 사용.
    """
    args_schema: Type[RelateComment_SearchToolArgsSchema] = RelateComment_SearchToolArgsSchema


    def loginCleverse(self):
        # 1. token 가져오기
        login_url = "https://hsi.cleverse.kr/api/auth/authenticate"
        login_headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }

        login_temp = {
            "userId":userId,
            "password":password,
            "domain":"hsi.cleverse.kr"
        }

        login_data = json.dumps(login_temp)

        login_response = requests.post(login_url, headers=login_headers, data=login_data)

        login_response_json = login_response.json()

        login_token = login_response_json.get("result").get("accessToken")
        
        return login_token

    def _run(self, articleId):
        
        commentList_url = "https://hsi.cleverse.kr/api/bbs/comment/commentList"
        
        
        login_token = self.loginCleverse()
        Authorization_text = "Bearer " + login_token#Bearer가 포함된 accessToken 입력
        
        commentList_headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Authorization": Authorization_text
        }

        # data
        commentList_temp = {
            "page":1,
            "size":100,
            "articleId":articleId,
            "boardId":125477,
            "boardMode":"COMPANY"
        }        
        
        commentList_data = json.dumps(commentList_temp)
    
        commentList_response = requests.post(commentList_url, headers=commentList_headers, data=commentList_data)

        commentList_response_json = commentList_response.json()
        
        
        commentList_response_df = pd.DataFrame(commentList_response_json)

        comments = commentList_response_df.result.commentList['content']


        commentTable = pd.DataFrame(columns=['commentId', 'content', 'nickName', 'commentDepth', 'orderGroup', 'articleId'])



        for i, comments_data in enumerate(comments) :
            commentTable = pd.concat([commentTable, pd.DataFrame([{'commentId' : comments_data['commentId'], 
                                                            'content' : comments_data['content'], 
                                                            'nickName' : comments_data['nickName'], 
                                                            'commentDepth' : comments_data['commentDepth'],
                                                            'orderGroup' : comments_data['orderGroup'],
                                                            'articleId' : comments_data['articleId']}])])
            # 게시글의 연관 comment 목록을 반환
        return commentTable    
        
        

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💼",
)



@st.cache_data(show_spinner="Searching CLEVERSE...")
def cleverse_search(query):

    # LLama 3.1 모델 설정
    llm = ChatOllama(
        model="mistral:latest",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,#.ZERO_SHOT_REACT_DESCRIPTION,#STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, #OPENAI_FUNCTIONS
        handle_parsing_errors=True,
        tools=[
            #Publication_GazetteTool(), 
            #Similar_SearchTool(),
            #RelateComment_SearchTool(),
            #CommentBased_ReplyTool(),
            BoardArticle_SearchTool(),
            BoardComment_SearchTool(),
        ],
        agent_kwargs={
            "system_message": SystemMessage(
                content="""
                당신은 사용자가 질문을 하면 게시판에서 유사한 게시글을 찾고 게시글의 articleId를 통해 게시글의 답글을 BoardComment_SearchTool을 통해 찾아서
                관련된 답글을 참고하여 창의적인 새로운 답변을 생성하여 사용자에게 인사이트를 제공해줍니다. 
                답변을 줄때는 단순히 설명하거나 요악하지 말고 사용자에게 당신이 알려주듯이 답변을 해야합니다.
                너는 사용자가 입력한 언어 그대로 검색해야 해. 특히, 한글로 입력된 키워드는 영어로 번역하지 않고 그대로 사용해야 한다.
            """
            #사용자의 입력에 대해 관련된 게시글을 찾고 게시글의 articleId를 통해 게시글의 답글을 BoardComment_SearchTool을 통해 찾아서 답글중에서 사용자 질의에 도움이 될만한 답글만을 선별하고 
                #분석하여 새로운 해석을 사용자에게 인사이트를 줄만한 새로운 답변을 리턴해준다."
                #  당신은 사용자의 문의사항에 답변을 생성하여 답변하는 챗봇입니다. 답변을 생성할때는 사용자의 질의사항 내용중 키워드를 뽑아서
                # 관련된 게시글을 찾고 찾은 게시글에 대한 articleId를 통해 답변글을 찾아서 답변 글들을 종합적으로 분석하여 이를바탕으로 새로운 독창적인 답변을 생성하여 리턴해주세요
                # 너는 사용자가 입력한 언어 그대로 검색해야 해. 특히, 한글로 입력된 키워드는 영어로 번역하지 않고 그대로 사용해야 한다.
                
                # 당신은 사용자가 질문을 하면 게시판에서 유사한 게시글을 찾고 관련된 답글을 랜덤으로 2~3개 정도 참고하여 창의적인 새로운 답변을 생성하여 사용자에게 인사이트를 제공해줍니다. 
                # 게시글의 articleId를 통해 게시글의 답글을 BoardComment_SearchTool을 통해 찾도록 한다.
                # 너는 사용자가 입력한 언어 그대로 검색해야 해. 특히, 한글로 입력된 키워드는 영어로 번역하지 않고 그대로 사용해야 한다.
                
            )
        },
    )

    #prompt = "검색단어 센서에 대한 재작년 특허 / 실용신안 공개 및 등록공보를 알려주세요."


    #input = "회의 에티튜드 관련글 있나요?"
    #"한화오션 주가 관련글 있나요? 답글들을 분석하여 새로운 독창적인 답변을 생성하여 리턴해주세요"
    #사용자가 입력한 내용에 대한 게시글을 찾고 해당글들의 답글들을 분석하여 새로운 독창적인 답변을 생성하여 리턴해주세요
    action = "질의에 관련된 글을 찾고 그 댓글을 찾아서 2~3개 정도만 참고하여 해당댓글 내용이 자신의 생각인 것처럼 저에게 조언주듯이 답변해주세요."
    #logininfo="loginID = "+loginID+"login password = " + passWord

    response = agent.invoke({"input": query+action})
            
    return response


@st.cache_data(show_spinner="Searching KIPRIS...")
def kipris_search(query):
    
    # LLama 3.1 모델 설정
    llm = ChatOllama(
        model="mistral:latest",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        tools=[
            Publication_GazetteTool(),
        ],
            agent_kwargs={
            "system_message": SystemMessage(
                content="""
                당신은 KIPRIS API 검색 전문가입니다..
                
                KIPRIS API 검색 결과에 대해 사용자에게 보기좋게 요약해주고
                
                약간의 인사이트도 추가로 제공헤줍니다.
            """
            )
        },
    )
    #prompt = "검색단어 센서에 대한 올해 특허 / 실용신안 공개 및 등록공보를 알려주세요."
    #with st.spinner("Searching KIPRIS API..."):
    response = agent.invoke(query)    
    return response


KIPRIS_REST_AccessKey=""
choice = "KIPRIS API"
query = ""
   

with st.sidebar:
    docs = None
    
    choice = st.selectbox(
        "Choose what you want to use", 
        (
            "KIPRIS API",
            "CLEVERSE API", 
        ),
    )
    
    
    if choice == "CLEVERSE API":        
        userId = st.text_input(
            "Input Your Clevese ID"
        )
        
        password = st.text_input(
            "Input Your Clevese PASSWORD",
            type = 'password'
        )        
    else:
        KIPRIS_REST_AccessKey = st.text_input(
            label = "Input KIPRIS API Key",
            value="sNuGpLPQtMYqZsEabQbu466205ZRjzXCDxgLpZMRdes="
        )
        
if choice == "CLEVERSE API":
    st.markdown(
        """
        # CLEVERSE GPT
                
        사이다봇 API.
        
        회사생활 관련 사이다봇에게 통해 문의하고자 하는 내용을 입력하세요.
        
        키워드 중심으로 궁금한 것일 적으시거나 궁금한 내용을 질문형태로 적으시면 됩니다.
        
        예) 회의 에티튜드 관련해서 알고 있나요?
        
        """
    )    
    
    query = st.text_input("Search Cleverse API")
    
    
    if query:
        # if not KIPRIS_REST_AccessKey: 
        #     st.write("좌측의 KIPRIS API Key를 입력하세요")
        # else:
        response = cleverse_search(query)
        #with st.spinner("Searching KIPRIS API..."):
        st.write(response["output"].replace("$", "\$"))
            


else:
    st.markdown(
    """
    # AI TSP GPT
            
    KIPRIS API.
    
    KIPRIS 특허 / 실용신안 공개 및 등록공보 검색용 GPT 입니다.
    
    특허 키워드와 검색년도(올해, 작년, 재작년 등)를 질문형태로 알려주세요
    
    입력 예) 검색단어 센서에 대한 올해 특허 / 실용신안 공개 및 등록공보를 알려주세요.
    
    """
    )
    
    query = st.text_input("Search KIPRIS API")

    if query:
        if not KIPRIS_REST_AccessKey: 
            st.write("좌측의 KIPRIS API Key를 입력하세요")
        else:
            response = kipris_search(query)
            # agent = initialize_agent(
            #     llm=llm,
            #     verbose=True,
            #     agent=AgentType.OPENAI_FUNCTIONS,
            #     handle_parsing_errors=True,
            #     tools=[
            #         Publication_GazetteTool(),
            #     ],
            # )

            # #prompt = "검색단어 센서에 대한 올해 특허 / 실용신안 공개 및 등록공보를 알려주세요."
            # #with st.spinner("Searching KIPRIS API..."):
            # response = agent.invoke(query)
            
            st.write(response["output"].replace("$", "\$"))





