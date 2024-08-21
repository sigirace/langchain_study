# 12.0 Introduction

- 회사이름을 agent에 입력 > agent는 검색엔진에서 search를 결정 > 회사 정보 얻고
- 사용자가 tool을 만들어주면 agent가 사용할 tool을 선택

# 12.1 Your First Agent

- LLM은 다음 token이 무엇일지 예측하는 애라서 연산등이 약한가봄
- func에는 str/function arg 다들어가면안되고 function name만
- agent type에서 Structed~ 얘가 tool의 input 갯수 여러개 사용 가능하게 함
  - agent type 살펴볼 필요
- langchain이 (prompt+우리가만든 tool)을 llm에게 전달> 받은 llm output을 langchain이 parsing(outputparser) > 사용자 PC에서 function 수행

# 12.2 How Do Agents Work

- 수행순서
  - agent는 llm에게 get_action(Json Blob)을 받아옴(agent finished를 받으면 끝 아니면 loop)
  - llm이 골라준 function을 수행(사용자 PC)하고 결과는 observation에 저장
  - 위 과정을 반복(repeat Thought,action,observation)
- llm에게 받은 함수호출, 함수 arg 세팅 등 모두 langchain의 역할

# 12.3 Zero-shot ReAct Agent

- ReAct는 Reasoning(Chain-of-Thought) + Acting(외부 도구 실행)
- gpt3/4 등을 사용하지 않을때, Zero-shot ReAct가 function calling을 지원하지 않는 다른 모델들과 같이 사용
- structured input react와의 차이점은 입력값 차이 stru~은 여러개,zero는 1개로 받아서 parsing 필요
- 필요시 agent를 LCEL을 사용해 커스텀 가능 but 딱히 그럴 필요 없음
- agent 고장 주 원인 : llm의 답이 langchain이 parsing 할 수 없는 형태로 답
  - handle_parsing_errors=True 이걸로 llm자체적으로 수정해서 던지나봐

# 12.4 OpenAI Functions Agent

- pydantic : 데이터 유효성 검증 라이브러리, 데이터가 어떤 형태여야 하는지 정의
- 장점
  - prompt 짧아져(llm에게 이전 과정들에 사용한 툴과 결과값을 기억할 필요 없음 ) 비용 절감
  - langchain이 함수 리스트를 formatting해서 llm에 전달(openai한정 동작한다는데 아님)

# 12.5 Search Tool

# 12.6 Stock Information Tools

- 매일/주간 주식 실적, 회사 관련 뉴스, 손익계산서등의 주식관련 많은 정보를 제공
- agent에 tool을 여러개 넣고 불러오게 하는 실습

# 12.8 SQLDatabaseToolkit

- toolkit.get_tools() : 툴킷의 각 툴에 대한 정보 전달
- 같은 agent에 파일의 툴킷을 더하고 database 툴킷을 더하는 식으로 agent 연동가능
- 쿼리 실행 중에 오류가 발생하면 SQL 에이전트가 문제를 식별하고 수정한 후 수정된 쿼리를 실행
- create_sql_agent는 두 가지 유형의 에이전트를 지원: OpenAI 함수 , ReAct 에이전트

# 12.9 Conclusions

- streamlit 배포
