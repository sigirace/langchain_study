# 12 InvestorGPT

## 12.0 Introduction
### agent에 대해서 알아보자. 우리만의 custom agent 만들고 custom tools를 사용 > investor GPT agent 제작
### sql database와 Chat하게 해주는 sqldatabase toolkit에 대해서도 알아보자


## 12.1 Your First Agent
### 첫번째 custom agent tool을 제작

prompt = "Cost of $355.39 + $924.87 + $721.2 + $1940.29 + $573.63 + $65.72 + $35.00 + $552.00 + $76.16 + $29.12"
llm.invoke(prompt)

llm이 정답을 출력해주진 않음 > 계산기가 더 정확함 > llm은 다음 텍스트 통계적 추론해주는 모델일뿐 계산기가 아님.

>> 우리가 agent를 만들고 agent를 위한 tool을 작성해주자.  > structuredTool을 사용하여 여러개의 input을 가질 수 있다.

```python
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType

def plus(a, b):
    return a + b


agent = initialize_agent(
    llm=llm,
    verbose=True, # 말이 많음 > 우리에게 agent가 하는 작업의 모든 과정 및 추론을 보여줌
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, # STRUCTURED_CHAT(chat에 최적화)_ZERO_SHOT_REACT(react논문에 기반한 agent)_DESCRIPTION
    tools=[
        StructuredTool.from_function(
            func=plus,
            name="Sum Calculator",
            description="Use this to perform sums of two numbers. This tool take two arguments, both  should be numbers.",
        ),
    ],
)

```



## 12.2 How Do Agents Work
### agent가 어떻게 작동하는지 설명

>> 사실상 어려운 일을 많이하는 것은 gpt3, gpt4보다는 langchain이다. > langchain 문서(https://python.langchain.com/v0.1/docs/modules/agents/concepts/))에 agent loop 가 어떻게 동작하는지 보여주는 pseudo code 부분이 있다.

LLM이 끝남을 의미하는 AgentFinish로 응답하지 않는다면 컴퓨터를 실행시킬 그 LLM이 고른 함수를 실행한다. > 이후 이전 next_action과 observation을 기반으로 다음 액션을 고르라고 할 것이다. 
> 위 과정울 LLM이 고른 action이 agent로 하여금 끝내라는 신호를 줄때까지 반복
![alt text](image.png)

>> langchain은 LLM의 응답을 가져와서 그 응답을 langchain이 파싱하고 응답을 파싱한 후에 langchain을 사용할 툴을 선택한다. langchain은 우리 컴퓨터에서 코드를 실행할 것이고 
langchain은 그 결과값을 가지고 다시 LLM에 돌려준다. 이 반복은 아래 응답이 올때까지 계속해서 반복한다.
![alt text](image-1.png)
위 응답은 langchain이 output parser을 활용해서 파싱한 결과이다.(Final Answer이 오면 멈춰야 할 시점을 알게된다.)
>> 가끔 LLM이 잘 동작하는 방법으로 응답하는 경우도 있어서 out parser에서 동작하지 않는 경우가 있음 > quiz에서 봤던 것처럼 그럴땐 우리가 LLM에게 가능한 이 형식으로 답변하라고 강요할 수 있다.
>> langchain에서 항상 동작하는 것처럼 보이는 엄청난 프롬프트를 만들었지만 아주 가끔 LLM은 json이 아닌 text로 응답할 때가 있는데 그럼 output parser가 LLM의 잘못된 답을 파싱할 수 없기 때문에
agent가 고장나는것 > OpenAI의 함수호출(Function call)기능을 활용가능함



## 12.3 Zero-shot ReAct Agent
### agent에 다양한 유형이 있지만 우리는 Zero-shot ReAct을 살펴본다., 12.4에서 배울 agent와는 달리 많은 모델들과 같이 사용가능
https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_functions_agent/

Zeroshot react : 가장 범용적인 목적의 agent
https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/
> 함수호출을 지원하는 OpenAI의 gpt3/4를 사용하지 않는다면 Zeroshot react가 함수 호출을 지원하지 않는 다른 모델들과 같이 사용하게 될 것이다.

Zeroshot ReAct와 Structured Input ReAct의 다른점은 Structured Input이 여러 입력을 가질 수 있지만 Zeroshot은 아니라는 것이다. 
> 아래와 같이 description에 적절한 문구를 추가하여 처리가능
description="Use this to perform sums of two numbers. Use this tool by sending a pair of number separated by a comma.\nExample:1,2"
> LLM이 우리가 원하는 출력 형식으로 응답하지 않아 output parser에 파싱에러가 발생하면 handle_parsing_errors=True를 통해 파싱에러를 고치도록 할 수 있다.(langchain이 자동으로 해결)

react논문(arxiv.org/pdf/2210.03629.pdf) : agent를 어떻게 활용하고 agent가 잘 동작하기 위해 어떤 지시를 줘야하는지 써있다.


## 12.4 OpenAI Functions Agent 
### GPT3/4 같은 OpanAI에서만 동작가능한 agent

>> Open AI 함수를 사용하기 위해선 다른 방법으로 Tool을 설정해야함 > Pydantic에 대해 알아보자 : python의 데이터 유효성 라이브러리중 하나(docs.pydantic.dev/latest)
> 우리의 데이터가 어떤 형태여야 하는지 알려준다. : OPENAI를 사용할 때 quizGPT에서 다루었던 것보다 훨씬 더 편리한 방법
> 프롬프트가 작으므로 비용을 아낄 수 있다. > OpenAI function agent로 pydantic을 사용하게 되면 입력값은 거의 없을 수 있다. >> quizgpt때 했던 json 포맷팅 작업 자동으로 해주고 있다.


## 12.5 Search Tool 
### 회사정보를 웹에서 찾는 툴 제작 > 회사가 상장했는지, 회사의 주식심볼(ticker)을 찾기


## 12.6 Stock Information Tools
### Alpha Vantage(alphavantage.co) 사용 : 가입후 API KEY 발급하여 일간/주간 주가실적 및 회사뉴스, 기타정보 확인가능 > 회사의 손익계산서를 주는 툴을 만들어보자
> 발급받은 키를 streamlit 폴더 안에 secrets.toml 파일에 넣기

> income-statement 참조 : https://www.alphavantage.co/documentation/#income-statement

해당예제에서 우리가 알고싶은 내용은 StockMarketSymbolSearchTool을 통해 Cloudflare의 주식심볼을 아는데 사용되어야 하고 CompanyOverviewTool 이건 정보를 얻는데 사용되어야 함
Tool 들을 정의하지만 LLM이 이 툴을 사용할거라는 보장은 없다. > LLM이 더 많은 정보가 필요하다고 느낄 때만 사용될 것이다.

>> OpenAPI 공식문서 [OpenAI overview](https://platform.openai.com/docs/models/overview) : gpt-3.5-turbo: 채팅에 특화된 모델
>> 실행하다 보면 문서가 과다하여 토큰초과 발생할 수 있음, 토큰 허용량 등 확인하여 모델을 변경


## 12.7 Agent Prompt 
### 06_InvestorGPT.py

agent에서 시스템 프롬프트를 바꾸는 방법?
OpenAI 함수 agent는 시스템 프롬프트에서 그냥 you are helpful AI assistance 라고만 말한다.
HUMAN 에서 실제 우리가 셋팅한 프롬프트가 나타난다.
![alt text](image-3.png)

시스템 프롬프트 자체를 당신은 주식거래자에요 라고 말해보자 > agent의 분위기를 설정. > agent_kwargs를 통해 셋팅함

GPT4 터보는 120,000 토근의 context window를 가지고 있어 토큰초과가 잘 뜨지 않음(유료)

alphavantage API가 하루에 25번정도 call 제한 있음 > 유료(무제한) : 한달 25달러




## 12.8 SQLDatabaseToolkit

langchain 의 장점은 아주 많은 서드파티 제공자와의 많은 병합이 가능하다는 것(https://python.langchain.com/v0.2/docs/integrations/tools/)
툴들을 agent에 그냥 추가만 하면 됨.

12.6에서 사용했던 alpha_vantage 도 langchain 유틸리티에 있음

![alt text](image-2.png)

>> toolkit.get_tools() : 툴깃의 각 툴에 대한 정보를 전달해줌


## 12.9 Conclusions
>> 스트림릿 클라우드를 통해 스트림릿 배포도 가능 > 무료이지만 퍼블릭에 오픈해야함




