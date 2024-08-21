# 9.4 Formatter Prompt

- example promtp는 Json형태로 리턴받아 UI에 쉽게 뿌려주기 위함
- prompt가 {{ 로 시작하는 이유는 llm이 formatting하면서 날려먹지 않게 하기 위해

# 9.6 Caching

- hash 한다는건 데이터에 서명을 하는것으로 동일한게 불려오면 재서명을 방지
- cache_data 이 함수가 함수를 살펴보며 hash값을 살피는 역할을 함
- \_를 붙이면(ex.\_doc) 서명을 변경하지않는다는 명시(streamlit기능,python x)
- \_는 문서를 항상 캐싱한다는 의미로 처음으로 들어온 변수로 고정
- 그래서 run_quiz_chain(\_doc,topic)으로 2번째 변수로 서명을 변경하게 세팅

# 9.7 Grading Questions

- form내에서는 streamlit의 특성인 리로딩이 안일어남,submit시에 한번만 일어남

# 9.8 Function Calling

- 함수를 llm이 호출
- create_quiz 함수를 llm이 인식하게 하는 방법 schema를 만들어준다
- function_call에 함수 여러개도 사용가능
- llm이 필요에따하 함수를 알아서 호출하게 하려면 function_call='auto'
- 함수가 존재하지 않아도 모델을 속여 원하는 형식(parameters를 통해)으로 제공 받을 수 있음
